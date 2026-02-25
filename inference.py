import os
import json
import glob
import torch
from model import Transformer, TransformerConfig, ByteTokenizer
from gpu_utils import set_device


@torch.no_grad()
def stream_generate(model, tokenizer, prompt, max_new_tokens=512*2*4, temperature=0.7, device='cpu'):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Store complete list for multi-byte decode stability
    full_ids = idx[0].tolist()
    last_decoded = tokenizer.decode(full_ids)

    scratchpad = None
    kv_caches = None
    for _ in range(max_new_tokens):
        logits, _, scratchpad, kv_caches = model(idx, scratchpad=scratchpad, kv_caches=kv_caches)
        logits_step = logits[:, -1, :] / temperature
        
        probs = torch.nn.functional.softmax(logits_step, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # In stream_generate, 'idx' becomes just the new token for efficiency
        idx = idx_next
        full_ids.append(idx_next.item())

        current_decoded = tokenizer.decode(full_ids)
        new_text = current_decoded[len(last_decoded):]
        if new_text:
            yield new_text
            last_decoded = current_decoded
            if "<eos>" in new_text:
                break


def main():
    device, _ = set_device(min_memory_gb=2.0)
    tokenizer = ByteTokenizer()
    
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384
    )
    
    model = Transformer(config).to(device)
    checkpoint_path = "model.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        # Prefer strict loading, but gracefully fall back to non-strict or fresh init
        try:
            state = torch.load(checkpoint_path, map_location=device)
            try:
                model.load_state_dict(state)
                print("Checkpoint loaded (strict match).")
            except RuntimeError as e_strict:
                print(f"\n[!] Strict load failed: {e_strict}")
                print("Attempting non-strict load (will ignore unexpected/missing keys)...")
                try:
                    model.load_state_dict(state, strict=False)
                    print("Loaded checkpoint with non-strict matching. Some parameters/buffers may be left at initialization values.")
                except Exception as e_nonstrict:
                    print(f"Non-strict load also failed: {e_nonstrict}")
                    print("Proceeding with newly initialized model (will not load checkpoint).")
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            print("Proceeding with newly initialized model.")
    else:
        print("Warning: model.pth not found. Generating with uninitialized weights.")

    while True:
        try:
            user_input = input("\nUser: ")
            prompt = f"<bos><role>user</role>{user_input}<role>assistant</role>"
            
            print("Assistant: ", end="", flush=True)
            for char in stream_generate(model, tokenizer, prompt, device=device):
                print(char, end="", flush=True)
                if char == "<eos>": break
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
