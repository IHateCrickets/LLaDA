import sys
import importlib
import json
import torch
from log_utils import log

# --- Configuration Loading ---
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
            config_trust_remote_code = model_config.get("trust_remote_code", False) # Default to False if missing

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
mlx_load = None

# --- Framework-Specific Logic & Loading ---
if framework == "torch":
    try:
        from generate import generate # Local generate function
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from configuration_llada import LLaDAConfig # Import local config class
        modeling_llada = importlib.import_module("modeling_llada")
        LLaDAModelLM = modeling_llada.LLaDAModelLM # Import local model class
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

        # --- Fallback Loading Logic ---
        model_loaded_successfully = False
        try:
            # Attempt 1: Try loading with AutoModelForCausalLM first
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

            # Refined check for incompatibility error
            if "data did not match" in error_str:
                log("WARNING", f"Caught specific incompatibility error. Attempting fallback with local LLaDAModelLM and force_download=True.")
                try:
                    # Attempt 2: Fallback to explicit local class loading with force_download
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
        # --- End Fallback Logic ---

        log("INFO", "PyTorch model loading process completed.")

    except ImportError as e:
        log("ERROR", f"Error importing PyTorch/Transformers components: {e}")
        log("ERROR", "Please ensure torch, transformers, modeling_llada.py, and configuration_llada.py are available.")
        sys.exit(1)
    except Exception as e: # Catch errors from loading attempts or the RuntimeError above
        log("ERROR", f"Error during PyTorch model loading sequence for '{config_model_id}': {e}")
        sys.exit(1)

    # --- PyTorch Chat Function ---
    def chat_pytorch():
        # Set defaults to 20
        gen_length = 20
        steps = 20
        block_length = 20

        print("Enter your questions below. Type 'quit' or 'exit' to end.")
        print('*' * 66)
        print(f'** PyTorch Mode | Model: {config_model_id} | Device: {device.upper()} **') # type: ignore
        print(f'** Answer Length: {gen_length} | Sampling Steps: {steps} | Block Length: {block_length} **') # Updated print
        print('*' * 66)

        conversation_num = 0
        prompt_history = None

        # Define progress callback for chat.py
        def print_progress(step, total_steps, block, total_blocks):
            # Calculate overall progress percentage
            current_total_step = (block - 1) * total_steps + step
            overall_total_steps = total_blocks * total_steps
            percentage = (current_total_step / overall_total_steps) * 100 if overall_total_steps > 0 else 0
            # Format progress bar
            bar_length = 20 # Length of the progress bar
            filled_length = int(bar_length * current_total_step // overall_total_steps)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            # Use carriage return to overwrite the line
            progress_str = f"Generating... [{bar}] {percentage:.1f}% (Ctrl+C to Cancel)"
            sys.stdout.write(f"\r{progress_str:<80}") # Pad to ensure line clears
            sys.stdout.flush()

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                if not user_input:
                    continue

                m = [{"role": "user", "content": user_input}]
                if hasattr(tokenizer, 'apply_chat_template'):
                     current_prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) # type: ignore
                else:
                     log("WARNING", "Tokenizer does not have apply_chat_template. Using raw input.")
                     current_prompt_str = user_input

                input_ids = tokenizer(current_prompt_str, return_tensors="pt")['input_ids'].to(device) # type: ignore

                if conversation_num == 0:
                    prompt_to_generate = input_ids
                else:
                    if prompt_history is None:
                         log("ERROR", "Prompt history is missing in subsequent turn.")
                         continue
                    bos_token_id = getattr(tokenizer, 'bos_token_id', None)
                    concat_ids = input_ids[:, 1:] if input_ids.shape[1] > 1 and bos_token_id is not None and input_ids[0, 0] == bos_token_id else input_ids
                    prompt_to_generate = torch.cat([prompt_history, concat_ids], dim=1) # type: ignore

                print("Bot:", end="", flush=True) # Print "Bot:" once before progress starts
                # Pass the callback to the generate function using updated defaults
                out = generate(model, prompt_to_generate, steps=steps, gen_length=gen_length,
                               block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence',
                               progress_callback=print_progress)
                # Clear the progress line after generation is complete
                sys.stdout.write('\r' + ' ' * 80 + '\r') # Overwrite progress line
                sys.stdout.flush()

                answer_ids = out[:, prompt_to_generate.shape[1]:]
                answer = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0] # type: ignore
                print(answer) # Print the final answer on a new line

                eos_token_id = getattr(tokenizer, 'eos_token_id', 126081)
                prompt_history = out[out != eos_token_id].unsqueeze(0).to(device)
                conversation_num += 1
                print('-----------------------------------------------------------------------')

            except KeyboardInterrupt:
                # Clear the progress line on interrupt
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()
                print("\nGeneration cancelled by user (Ctrl+C).")
                prompt_history = None # Reset history to avoid issues on next turn
                conversation_num = 0 # Reset conversation context
                print('-----------------------------------------------------------------------')
                continue # Go to next input prompt
            except Exception as e:
                # Clear the progress line on error
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()
                log("ERROR", f"An error occurred during PyTorch chat: {e}")
                print(f"\nAn error occurred: {e}")
                print('-----------------------------------------------------------------------')

    # --- PyTorch Main Execution ---
    if __name__ == "__main__":
        chat_pytorch()

elif framework == "mlx":
    try:
        from mlx_lm import load as mlx_load_import, generate as mlx_generate_import # type: ignore
        mlx_generate = mlx_generate_import
        mlx_load = mlx_load_import
        log("INFO", "Using Framework: MLX")
        log("INFO", "Note: MLX performance is best on Apple Silicon.")
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

    # --- MLX Chat Function ---
    def chat_mlx():
        gen_length = 32
        temperature = 0.0
        print("Enter your questions below. Type 'quit' or 'exit' to end.")
        print('*' * 66)
        print(f'** MLX Mode | Model: {config_model_id} **')
        print(f'** Answer Length: {gen_length} **')
        print('*' * 66)

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                if not user_input:
                    continue

                messages = [{"role": "user", "content": user_input}]

                if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None: # type: ignore
                     prompt = tokenizer.apply_chat_template(  #type: ignore
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                     log("WARNING", "Tokenizer does not have a chat template. Using raw input.")
                     prompt = user_input

                print("Bot:", end="", flush=True)
                # MLX generate streams with verbose=True, cancellation via Ctrl+C is handled by Python
                response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=gen_length, temp=temperature, verbose=True) # type: ignore
                print()
                print('-----------------------------------------------------------------------')

            except KeyboardInterrupt:
                print("\nGeneration cancelled by user (Ctrl+C).")
                print('-----------------------------------------------------------------------')
                continue # Go to next input prompt
            except Exception as e:
                log("ERROR", f"An error occurred during MLX chat: {e}")
                print(f"\nAn error occurred: {e}")
                print('-----------------------------------------------------------------------')

    # --- MLX Main Execution ---
    if __name__ == "__main__":
        chat_mlx()

else:
    log("ERROR", f"Error: Unknown framework '{framework}' specified in {CONFIG_FILE} or defaults.")
    sys.exit(1)
