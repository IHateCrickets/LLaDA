import torch
import numpy as np
import torch.nn.functional as F
import importlib # Added for dynamic import in main
import sys # Added for progress callback
from typing import Optional, Callable # Added for typing

from transformers import AutoTokenizer, AutoModel

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise + 1e-9)) ** temperature
    return logits.exp() / (gumbel_noise + 1e-9)

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    # Ensure steps is at least 1 to avoid division by zero
    steps = max(1, int(steps)) # Ensure steps is an integer >= 1

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        # Ensure remainder[i] is not negative if mask_num is 0
        rem_val = remainder[i].item() # Get Python int
        if rem_val > 0:
             num_transfer_tokens[i, :rem_val] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             progress_callback: Optional[Callable[[int, int, int, int], None]] = None): # Added callback parameter
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        progress_callback: Optional function to call with (current_step, total_steps, current_block, total_blocks).
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    # Ensure block_length is valid and calculate num_blocks correctly
    if block_length <= 0: block_length = gen_length
    block_length = min(block_length, gen_length)
    num_blocks = (gen_length + block_length - 1) // block_length # Ceiling division

    # Ensure steps is valid and calculate steps_per_block correctly
    steps = max(1, int(steps))
    steps_per_block = steps // num_blocks
    if steps_per_block < 1: steps_per_block = 1
    total_effective_steps = steps_per_block * num_blocks # Total steps that will actually run

    current_total_step = 0 # Track overall step count for callback if needed, though per-block is clearer

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = min(prompt.shape[1] + (num_block + 1) * block_length, x.shape[1])
        block_mask_index = (x[:, block_start:block_end] == mask_id)

        if not block_mask_index.any(): continue # Skip block if no masks

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            if progress_callback:
                progress_callback(i + 1, steps_per_block, num_block + 1, num_blocks)

            mask_index = (x == mask_id)
            if not mask_index.any(): break

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                model_outputs = model(input_ids=x_)
                logits = getattr(model_outputs, 'logits', model_outputs[0])
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                model_outputs = model(input_ids=x)
                logits = getattr(model_outputs, 'logits', model_outputs[0])

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
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

            x[transfer_index] = x0[transfer_index]

    if progress_callback:
         sys.stdout.write('\r' + ' ' * 80 + '\r')
         sys.stdout.flush()
    
    return x

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use LLaDAModelLM for loading if using the original model
    # Assuming generate.py might be run standalone, use AutoModel as fallback
    try:
        modeling_llada = importlib.import_module("modeling_llada")
        LLaDAModelLM = modeling_llada.LLaDAModelLM
        model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    except Exception:
        print("Warning: Could not load custom LLaDAModelLM. Falling back to AutoModel. Ensure modeling_llada.py is present for full functionality.")
        model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device) # type: ignore

    # Define a simple progress printer for standalone testing
    def print_standalone_progress(step, total_steps, block, total_blocks):
        progress = f"Block {block}/{total_blocks}, Step {step}/{total_steps}"
        sys.stdout.write(f"\rGenerating... {progress}")
        sys.stdout.flush()

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', progress_callback=print_standalone_progress)
    print() # Newline after progress
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
