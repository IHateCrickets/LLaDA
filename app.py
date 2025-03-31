import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
import time
import re
import sys
import os # Added for os._exit
import importlib
import json
import logging
from log_utils import log
import webbrowser
import threading

# --- Configuration Loading ---
# ... (Config loading code remains the same) ...
CONFIG_FILE = "model_config.json"
DEFAULT_FRAMEWORK = "torch"
DEFAULT_MODEL_KEY = "default_torch"
DEFAULT_MODEL_CONFIG = {
    "framework": DEFAULT_FRAMEWORK,
    "model_id": "GSAI-ML/LLaDA-8B-Instruct",
    "trust_remote_code": True
}

config_framework = DEFAULT_FRAMEWORK
config_model_id = DEFAULT_MODEL_CONFIG["model_id"]
config_trust_remote_code = DEFAULT_MODEL_CONFIG["trust_remote_code"]

try:
    with open(CONFIG_FILE, 'r') as f:
        config_data = json.load(f)
        active_key = config_data.get("active_model_key", DEFAULT_MODEL_KEY)
        available_models = config_data.get("available_models", {})

        if active_key in available_models:
            model_config = available_models[active_key]
            config_framework = model_config.get("framework", DEFAULT_FRAMEWORK).lower()
            config_model_id = model_config.get("model_id")
            config_trust_remote_code = model_config.get("trust_remote_code", False)

            if not config_model_id:
                log("ERROR", f"Missing 'model_id' for active key '{active_key}' in {CONFIG_FILE}. Using defaults.")
                config_framework = DEFAULT_FRAMEWORK
                config_model_id = DEFAULT_MODEL_CONFIG["model_id"]
                config_trust_remote_code = DEFAULT_MODEL_CONFIG["trust_remote_code"]
            else:
                 log("INFO", f"Loaded configuration from {CONFIG_FILE} for key '{active_key}': framework='{config_framework}', model_id='{config_model_id}', trust_remote_code={config_trust_remote_code}")

        else:
            log("ERROR", f"Active model key '{active_key}' not found in 'available_models' in {CONFIG_FILE}. Using defaults.")
            config_framework = DEFAULT_FRAMEWORK
            config_model_id = DEFAULT_MODEL_CONFIG["model_id"]
            config_trust_remote_code = DEFAULT_MODEL_CONFIG["trust_remote_code"]

except FileNotFoundError:
    log("WARNING", f"{CONFIG_FILE} not found. Using default model '{DEFAULT_MODEL_CONFIG['model_id']}'.")
    config_framework = DEFAULT_FRAMEWORK
    config_model_id = DEFAULT_MODEL_CONFIG["model_id"]
    config_trust_remote_code = DEFAULT_MODEL_CONFIG["trust_remote_code"]
except json.JSONDecodeError:
    log("ERROR", f"Error decoding {CONFIG_FILE}. Using default model '{DEFAULT_MODEL_CONFIG['model_id']}'.")
    config_framework = DEFAULT_FRAMEWORK
    config_model_id = DEFAULT_MODEL_CONFIG["model_id"]
    config_trust_remote_code = DEFAULT_MODEL_CONFIG["trust_remote_code"]
except Exception as e:
    log("ERROR", f"Error loading {CONFIG_FILE}: {e}. Using default model '{DEFAULT_MODEL_CONFIG['model_id']}'.")
    config_framework = DEFAULT_FRAMEWORK
    config_model_id = DEFAULT_MODEL_CONFIG["model_id"]
    config_trust_remote_code = DEFAULT_MODEL_CONFIG["trust_remote_code"]

# --- Global Variables ---
model = None
tokenizer = None
device = None
framework = config_framework
mlx_generate = None
demo_instance = None

# --- Helper Function Definitions ---
def add_gumbel_noise(logits, temperature):
    if temperature <= 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise + 1e-9)) ** temperature
    return logits.exp() / (gumbel_noise + 1e-9)

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = max(1, int(steps))
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        rem_val = remainder[i].item()
        if rem_val > 0: num_transfer_tokens[i, :rem_val] += 1
    return num_transfer_tokens

# --- Framework-Specific Loading ---
if framework == "torch":
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from configuration_llada import LLaDAConfig
        modeling_llada = importlib.import_module("modeling_llada")
        LLaDAModelLM = modeling_llada.LLaDAModelLM
        log("INFO", "Using Framework: PyTorch")

        if torch.cuda.is_available():
            device = 'cuda'
            dtype = torch.bfloat16
            log("INFO", "CUDA detected. Using GPU with bfloat16.")
        else:
            device = 'cpu'
            dtype = torch.float32
            log("INFO", "CUDA not available. Using CPU with float32.")

        log("INFO", f"Loading PyTorch model: {config_model_id}...")
        assert isinstance(config_model_id, str), f"Model ID must be a string, got: {type(config_model_id)}"
        tokenizer = AutoTokenizer.from_pretrained(config_model_id, trust_remote_code=config_trust_remote_code)

        model_loaded_successfully = False
        try:
            log("INFO", f"Attempt 1: Loading model {config_model_id} using AutoModelForCausalLM with trust_remote_code={config_trust_remote_code}")
            model = AutoModelForCausalLM.from_pretrained(
                config_model_id,
                trust_remote_code=config_trust_remote_code,
                torch_dtype=dtype
            ).to(device).eval()
            log("INFO", "Attempt 1: Successfully loaded model using AutoModelForCausalLM.")
            model_loaded_successfully = True

        except Exception as e1:
            error_str = str(e1).lower()
            log("WARNING", f"Attempt 1 failed: {e1}")
            log("DEBUG", f"Attempt 1 error string (lower): {error_str}")

            if "data did not match" in error_str:
                log("WARNING", f"Caught specific incompatibility error. Attempting fallback with local LLaDAModelLM and force_download=True.")
                try:
                    log("INFO", f"Attempt 2: Loading config explicitly for {config_model_id} using local LLaDAConfig with trust_remote_code={config_trust_remote_code}, force_download=True")
                    loaded_config = LLaDAConfig.from_pretrained(
                        config_model_id,
                        trust_remote_code=config_trust_remote_code,
                        force_download=True
                    )
                    log("INFO", f"Attempt 2: Loading model explicitly for {config_model_id} using local LLaDAModelLM with trust_remote_code={config_trust_remote_code}, force_download=True")
                    model = LLaDAModelLM.from_pretrained(
                        config_model_id,
                        config=loaded_config,
                        trust_remote_code=config_trust_remote_code,
                        force_download=True,
                        torch_dtype=dtype
                    ).to(device).eval()
                    log("INFO", "Attempt 2: Successfully loaded model using fallback LLaDAModelLM.")
                    model_loaded_successfully = True
                except Exception as e2:
                    log("ERROR", f"Attempt 2 (Fallback) loading with LLaDAModelLM also failed: {e2}")
            else:
                log("ERROR", f"Attempt 1 failed with an unexpected error (not triggering fallback): {e1}")

        if not model_loaded_successfully or model is None:
             raise RuntimeError(f"Failed to load PyTorch model '{config_model_id}' after all attempts.")

        log("INFO", "PyTorch model loading process completed.")

    except ImportError as e:
        log("ERROR", f"Error importing PyTorch/Transformers components: {e}")
        log("ERROR", "Please ensure torch, transformers, modeling_llada.py, and configuration_llada.py are available.")
        sys.exit(1)
    except Exception as e:
        log("ERROR", f"Error during PyTorch model loading sequence for '{config_model_id}': {e}")
        sys.exit(1)

elif framework == "mlx":
    try:
        from mlx_lm import load as mlx_load_import, generate as mlx_generate_import # type: ignore
        mlx_generate = mlx_generate_import
        mlx_load = mlx_load_import
        log("INFO", "Using Framework: MLX")
        log("INFO", "Note: MLX performance is best on Apple Silicon. Visualization will be limited.")
        device = None

        log("INFO", f"Loading MLX model: {config_model_id}...")
        assert isinstance(config_model_id, str), f"Model ID must be a string, got: {type(config_model_id)}"
        model, tokenizer = mlx_load(config_model_id)
        log("INFO", "MLX model loaded successfully.")

    except ImportError:
        log("ERROR", "Error: 'mlx-lm' library not found.")
        log("ERROR", "MLX framework requires 'mlx-lm'. Installation failed previously due to potential OS incompatibility or dependency conflicts.")
        log("ERROR", "Please try 'pip install mlx-lm' manually if you are on macOS with Apple Silicon.")
        sys.exit(1)
    except Exception as e:
        log("ERROR", f"Error loading MLX model '{config_model_id}': {e}")
        log("ERROR", "Ensure the model ID is correct and you have network access.")
        log("ERROR", "Also confirm 'mlx-lm' and its dependencies (like 'mlx') are correctly installed.")
        sys.exit(1)

else:
    log("ERROR", f"Error: Unknown framework '{framework}' specified in {CONFIG_FILE} or defaults.")
    sys.exit(1)

# --- Constants ---
MASK_TOKEN = "[MASK]"
MASK_ID = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else 126336
log("INFO", f"Using MASK_ID: {MASK_ID}")

# --- Helper Functions (Parsing) ---
def parse_constraints(constraints_text):
    constraints = {}
    if not constraints_text: return constraints
    parts = constraints_text.split(',')
    for part in parts:
        if ':' not in part: continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
            word = word.strip()
            if word and pos >= 0: constraints[pos] = word
        except ValueError: continue
    return constraints

def format_chat_history(history):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg: messages.append({"role": "assistant", "content": assistant_msg})
    return messages

# --- Generation Logic ---
# ... (generate_response_with_visualization remains the same) ...
def generate_response_with_visualization(framework, model, tokenizer, device, messages, gen_length=64, steps=32,
                                         constraints=None, temperature=0.0, cfg_scale=0.0, block_length=32,
                                         remasking='low_confidence'):
    global MASK_ID
    yield "Starting generation...", [], None

    if framework == "torch":
        try:
            yield "Processing constraints...", [], None
            if constraints is None: constraints = {}
            processed_constraints = {}
            for pos, word in constraints.items():
                tokens = tokenizer.encode(" " + word, add_special_tokens=False)
                for i, token_id in enumerate(tokens):
                    processed_constraints[pos + i] = token_id

            yield "Preparing prompt...", [], None
            log("DEBUG", f"Messages before apply_chat_template (PyTorch): {messages}")
            chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(chat_input, return_tensors="pt")['input_ids'].to(device)

            prompt_length = input_ids.shape[1]
            x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
            x[:, :prompt_length] = input_ids.clone()

            initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
            yield "Initializing sequence...", initial_state, None

            for pos, token_id in processed_constraints.items():
                absolute_pos = prompt_length + pos
                if absolute_pos < x.shape[1]:
                    x[:, absolute_pos] = token_id

            prompt_index = (x != MASK_ID)

            if block_length > gen_length: block_length = gen_length
            num_blocks = (gen_length + block_length - 1) // block_length
            steps_per_block = steps // num_blocks
            if steps_per_block < 1: steps_per_block = 1

            for num_block in range(num_blocks):
                block_start = prompt_length + num_block * block_length
                block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
                block_mask_index = (x[:, block_start:block_end] == MASK_ID)

                if not block_mask_index.any(): continue

                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    yield f"Running Block {num_block+1}/{num_blocks}, Step {i+1}/{steps_per_block}...", None, None
                    mask_index = (x == MASK_ID)
                    if not mask_index.any(): break

                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = MASK_ID
                        x_ = torch.cat([x, un_x], dim=0)
                        model_outputs = model(input_ids=x_)
                        logits = getattr(model_outputs, 'logits', model_outputs[0])
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        model_outputs = model(input_ids=x)
                        logits = getattr(model_outputs, 'logits', model_outputs[0])

                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    if remasking == 'low_confidence':
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    elif remasking == 'random':
                        x0_p = torch.rand_like(x0, dtype=torch.float)
                    else:
                        raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

                    x0_p[:, block_end:] = -float('inf')
                    old_x = x.clone()
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -float('inf'))

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        block_confidence = confidence[j, block_start:block_end]
                        current_k = num_transfer_tokens[j, i].item()
                        k_safe = min(current_k, block_confidence.numel())
                        if k_safe <= 0: continue

                        if i < steps_per_block - 1:
                            _, select_indices = torch.topk(block_confidence, k=int(k_safe))
                            select_indices = select_indices + block_start
                            transfer_index[j, select_indices] = True
                        else:
                            transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]

                    x = torch.where(transfer_index, x0, x)

                    for pos, token_id in processed_constraints.items():
                        absolute_pos = prompt_length + pos
                        if absolute_pos < x.shape[1]:
                            x[:, absolute_pos] = token_id

                    current_state = []
                    for i_vis in range(gen_length):
                        pos = prompt_length + i_vis
                        token_id_vis = x[0, pos].item()
                        old_token_id_vis = old_x[0, pos].item()

                        if token_id_vis == MASK_ID:
                            current_state.append((MASK_TOKEN, "#444444"))
                        elif old_token_id_vis == MASK_ID:
                            token = tokenizer.decode([token_id_vis], skip_special_tokens=True)
                            confidence_val = float(x0_p[0, pos].cpu())
                            color = "#66CC66"
                            if confidence_val < 0.3: color = "#FF6666"
                            elif confidence_val < 0.7: color = "#FFAA33"
                            current_state.append((token, color))
                        else:
                            token = tokenizer.decode([token_id_vis], skip_special_tokens=True)
                            current_state.append((token, "#6699CC"))
                    yield None, current_state, None

            response_tokens = x[0, prompt_length:]
            final_text = tokenizer.decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            yield "Generation complete.", None, final_text

        except Exception as e:
             error_msg = f"Error during PyTorch generation: {str(e)}"
             print(error_msg)
             log("ERROR", error_msg)
             yield error_msg, [(error_msg, "red")], error_msg


    elif framework == "mlx":
        try:
            yield "Generating with MLX...", [], None
            log("DEBUG", f"Messages before apply_chat_template (MLX): {messages}")
            chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if constraints:
                print("Warning: Constraints are not directly applied in MLX generation via mlx_lm.")

            response_text = mlx_generate(model, tokenizer, prompt=chat_input, max_tokens=gen_length, temp=temperature if temperature > 0 else 0.0, verbose=False) # type: ignore

            final_state = []
            if tokenizer:
                final_tokens = tokenizer.encode(response_text, add_special_tokens=False) # type: ignore
                for i in range(gen_length):
                    if i < len(final_tokens):
                         token_str = tokenizer.decode([final_tokens[i]], skip_special_tokens=True) # type: ignore
                         final_state.append((token_str, "#6699CC"))
                    else:
                         final_state.append((MASK_TOKEN, "#444444"))
            else:
                final_state = [(response_text, "#6699CC")]

            yield "Generation complete (MLX).", final_state, response_text

        except Exception as e:
             error_msg = f"Error during MLX generation: {str(e)}"
             print(error_msg)
             log("ERROR", error_msg)
             yield error_msg, [(error_msg, "red")], error_msg

    else:
        raise ValueError(f"Unsupported framework in generate_response: {framework}")

# --- Gradio UI ---
css = '''
.category-legend{display:none}
button{height: 60px}
'''
def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# LLaDA - Large Language Diffusion Model Demo")
        device_label = "CPU"
        if framework == "torch" and device == "cuda":
            device_label = "CUDA GPU"
        elif framework == "mlx":
            device_label = "MLX (CPU/GPU)"
        loaded_model_id = config_model_id
        gr.Markdown(f"Using Framework: **{framework.upper()}** | Running on: **{device_label}** | Model: **{loaded_model_id}**")
        gr.Markdown("[model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), [project page](https://ml-gsai.github.io/LLaDA-demo/)")

        chat_history = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", height=450)
                with gr.Group():
                    with gr.Row():
                        user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", show_label=False, scale=3)
                        send_btn = gr.Button("Send", scale=1)
                        cancel_btn = gr.Button("Cancel", scale=1) # Added cancel button
                constraints_input = gr.Textbox(label="Word Constraints", info="PyTorch Only: 'position:word' format. Example: '0:Once, 5:upon, 10:time'", placeholder="0:Once, 5:upon, 10:time", value="")
                progress_label = gr.Label(label="Status", value="Idle.")
            with gr.Column(scale=2):
                output_vis = gr.HighlightedText(label="Denoising Process Visualization (PyTorch Only)", combine_adjacent=False, show_legend=True)

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(minimum=16, maximum=128, value=64, step=8, label="Generation Length")
                steps = gr.Slider(minimum=8, maximum=64, value=16, step=4, label="Denoising Steps (PyTorch Only)")
            with gr.Row():
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
                cfg_scale = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="CFG Scale (PyTorch Only)")
            with gr.Row():
                block_length = gr.Slider(minimum=8, maximum=128, value=32, step=8, label="Block Length (PyTorch Only)")
                remasking_strategy = gr.Radio(choices=["low_confidence", "random"], value="low_confidence", label="Remasking Strategy (PyTorch Only)")
            with gr.Row():
                visualization_delay = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.05, label="Visualization Delay (PyTorch Only, seconds)")


        with gr.Row():
            clear_btn = gr.Button("Clear Conversation")
            shutdown_btn = gr.Button("Shutdown Server")

        # HELPER FUNCTIONS
        def add_message(history, message, response):
            history = history.copy()
            history.append([message, response])
            return history

        def user_message_submitted(message, history):
            if not message.strip():
                return history, history, "", [], "Idle."
            history = add_message(history, message, None)
            history_for_display = history.copy()
            message_out = ""
            return history, history_for_display, message_out, [], "Generating..."

        def bot_response_wrapper(history, gen_len, steps_val, constraints, delay, temp, cfg, block_len, remasking):
            final_text_output = ""
            if not history or history[-1][1] is not None:
                 last_vis_state = []
                 final_text_output = history[-1][1] if history and history[-1][1] else ""
                 if history and final_text_output and tokenizer:
                     try:
                         log("DEBUG", f"History before format (reconstruct): {history}")
                         messages_reconstruct = format_chat_history(history)
                         log("DEBUG", f"Messages after format (reconstruct): {messages_reconstruct}")
                         final_tokens = tokenizer.encode(final_text_output, add_special_tokens=False) # type: ignore
                         for i in range(gen_len):
                             if i < len(final_tokens):
                                 token_str = tokenizer.decode([final_tokens[i]], skip_special_tokens=True) # type: ignore
                                 last_vis_state.append((token_str, "#6699CC"))
                             else:
                                 last_vis_state.append((MASK_TOKEN, "#444444"))
                     except Exception as e:
                         log("ERROR", f"Error reconstructing final state: {e}")
                         last_vis_state = [(final_text_output, "#6699CC")]
                 yield history, last_vis_state, "Idle."
                 return

            last_user_message = history[-1][0]
            try:
                log("DEBUG", f"History before format (generation): {history}")
                messages = format_chat_history(history[:-1])
                messages.append({"role": "user", "content": last_user_message})
                log("DEBUG", f"Messages after format (generation): {messages}")
                parsed_constraints = parse_constraints(constraints) if framework == "torch" else {}

                response_generator = generate_response_with_visualization(
                    framework, model, tokenizer, device,
                    messages,
                    gen_length=gen_len,
                    steps=steps_val,
                    constraints=parsed_constraints,
                    temperature=temp,
                    cfg_scale=cfg,
                    block_length=block_len,
                    remasking=remasking
                )

                vis_state = []
                progress_text = "Starting..."
                for update in response_generator:
                    progress_update, vis_update, text_update = update
                    if progress_update is not None: progress_text = progress_update
                    if vis_update is not None: vis_state = vis_update
                    if text_update is not None:
                        final_text_output = text_update
                        history[-1][1] = final_text_output
                    yield history, vis_state, progress_text

            except gr.Error as e:
                 if "process cancelled" in str(e).lower():
                     log("INFO", "Generation cancelled by user.")
                     if history and history[-1][1] is None:
                         history[-1][1] = "[Cancelled]"
                     yield history, [], "Cancelled."
                 else:
                     log("ERROR", f"Gradio error during generation: {e}")
                     yield history, [], f"Gradio Error: {e}"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                log("ERROR", f"Error in bot_response_wrapper: {error_msg}")
                error_vis = [(error_msg, "red")]
                try:
                    history[-1][1] = error_msg
                except IndexError: pass
                yield history, error_vis, "Error!"


        def clear_conversation():
            return [], [], [], "Idle."

        def shutdown_server():
            log("INFO", "Shutdown requested via UI button.")
            print("Attempting server shutdown...")
            status_update = gr.update(value="Shutting down...")
            threading.Timer(0.5, lambda: os._exit(0)).start()
            return status_update

        def cancel_triggered():
             log("INFO", "Cancel button clicked.")
             return gr.update(value="Cancelling...")

        # EVENT HANDLERS
        clear_btn.click(fn=clear_conversation, inputs=[], outputs=[chat_history, chatbot_ui, output_vis, progress_label])
        shutdown_btn.click(fn=shutdown_server, inputs=None, outputs=[progress_label])

        trigger_inputs = [
             chat_history, gen_length, steps, constraints_input,
             visualization_delay, temperature, cfg_scale, block_length, remasking_strategy
        ]
        response_outputs = [chatbot_ui, output_vis, progress_label]

        # Link Submit/Send to Bot Response and make them cancellable by cancel_btn
        submit_event = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input, output_vis, progress_label]
        ).then(
            fn=bot_response_wrapper,
            inputs=trigger_inputs,
            outputs=response_outputs,
            show_progress="hidden"
        )

        send_event = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input, output_vis, progress_label]
        ).then(
            fn=bot_response_wrapper,
            inputs=trigger_inputs,
            outputs=response_outputs,
            show_progress="hidden"
        )

        # Cancel button cancels the submit and send events AND updates status
        cancel_btn.click(fn=cancel_triggered, inputs=None, outputs=[progress_label], cancels=[submit_event, send_event])

    return demo

# --- Browser Launch Logic ---
def open_browser(url):
    log("INFO", f"Attempting to open browser at {url} as fallback...")
    try:
        webbrowser.open(url)
    except Exception as e:
        log("ERROR", f"Failed to open browser automatically: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    if model is None or tokenizer is None:
        print("Error: Model or Tokenizer failed to load. Exiting.")
        log("CRITICAL", "Model or Tokenizer failed to load. Exiting.")
        sys.exit(1)

    demo_instance = create_chatbot_demo()
    fallback_timer = None

    try:
        log("INFO", "Launching Gradio app...")
        default_local_url = "http://127.0.0.1:7860"
        fallback_timer = threading.Timer(7.0, open_browser, args=[default_local_url])
        fallback_timer.start()

        demo_instance.queue().launch(share=True, inbrowser=True)

        if fallback_timer is not None and fallback_timer.is_alive():
             fallback_timer.cancel()
        log("INFO", "Gradio app shut down.")

    except Exception as e:
        log("CRITICAL", f"Failed to launch Gradio app: {e}")
        if fallback_timer is not None and fallback_timer.is_alive():
            fallback_timer.cancel()
        sys.exit(1)
