import os
import torch
import sys
from model import NeuxbaneSSM, BPETokenizer

# Configuration
MODEL_PATH = "checkpoint/neuxbane_thinking_125m"
MAX_NEW_TOKENS = 512 # BPE is more efficient, can generate more
TEMPERATURE = 0.7

def main():
    # Use Cuda 1 if available (often faster or chosen by user)
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing on {device}...")

    # 1. Initialize standard GPT-2 BPE Tokenizer
    tokenizer = BPETokenizer()

    # 2. Initialize Model (GPT2-BPE-based 125M Architecture)
    model = NeuxbaneSSM()
    
    # 3. Load Model weights if exist
    weights_path = MODEL_PATH + ".pth"
    if os.path.exists(weights_path):
        print(f"[*] Loading weights from {weights_path}...")
        # Checkpoint might be bfloat16 or float32, map correctly
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"[!] Warning: No checkpoint found at {weights_path}. Using random weights.")
    
    model.to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()

    print("\n" + "="*50)
    print(" NEUXBANE THINKING (CUSTOM SSM + SCRATCHPAD - 125M BYTES)")
    print(" Type your prompt and press Enter. Ctrl+C to exit.")
    print("="*50 + "\n")

    # Persistent Dynamic Scratchpad for long-term conversation memory (Right-Brain)
    memory = None

    while True:
        try:
            prompt = input("User> ").strip()
            if not prompt: continue
            
            # Format prompt matching the exact tokens in train.py
            # [<user>] [198] [content] [198] [<assistant>] [198]
            nl_token_id = tokenizer.tokenizer.encode("\n", add_special_tokens=False)[0]
            
            input_ids_list = [tokenizer.user_token_id, nl_token_id]
            input_ids_list.extend(tokenizer.tokenizer.encode(prompt, add_special_tokens=False))
            input_ids_list.extend([nl_token_id, tokenizer.assistant_token_id, nl_token_id])
            
            input_ids = torch.LongTensor(input_ids_list).unsqueeze(0).to(device)
            
            print("Assistant> ", end="", flush=True)
            
            # Efficient decoding: Use KV-cache and positional indexing
            cache_params = None
            total_seen = 0
            current_input = input_ids
            generated_ids = input_ids # Initialize generated_ids
            
            with torch.no_grad():
                for i in range(MAX_NEW_TOKENS):
                    S = current_input.shape[1]
                    pos = torch.arange(total_seen, total_seen + S, device=device)
                    
                    logits, cache_params, memory, _ = model(
                        input_ids=current_input,
                        memory=memory,
                        cache_params=cache_params,
                        use_cache=True,
                        cache_position=pos
                    )
                    
                    total_seen += S
                    
                    next_token_logits = logits[:, -1, :] / max(TEMPERATURE, 1e-6)
                    
                    # Apply penalty to avoid loops
                    if i > 0:
                         window = generated_ids[0, -30:]
                         for t in window:
                             next_token_logits[0, t] -= 0.6
                            
                    # Numerical stability: clamp logits
                    next_token_logits = torch.clamp(next_token_logits, -50, 50)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # Safety check for NaN/Inf
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        print("\n[!] Error: Model produced unstable probabilities (NaN/Inf).")
                        break
                        
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    # Next step only processes the single new token
                    current_input = next_token
                    
                    # Decode and stream
                    char_token = next_token.item()
                    if char_token == tokenizer.eos_token_id:
                        break
                    
                    print(tokenizer.decode([char_token]), end="", flush=True)
                    
            print("\n")
                    
            print("\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[!] Error during inference: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
