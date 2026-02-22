import os
import torch
import sys
from model import NeuxbaneThinking, BPETokenizer

# Configuration
MODEL_PATH = "checkpoint/neuxbane_thinking_125m"
MAX_NEW_TOKENS = 512 # BPE is more efficient, can generate more
TEMPERATURE = 0.7

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing on {device}...")

    # 1. Initialize standard GPT-2 BPE Tokenizer
    tokenizer = BPETokenizer()

    # 2. Initialize Model (GPT2-BPE-based 125M Architecture)
    model = NeuxbaneThinking()
    
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
    print(" NEUXBANE THINKING (MAMBA SSM + SCRATCHPAD - 125M BYTES)")
    print(" Type your prompt and press Enter. Ctrl+C to exit.")
    print("="*50 + "\n")

    # Persistent Dynamic Scratchpad for long-term conversation memory (Right-Brain)
    memory = None

    while True:
        try:
            prompt = input("User> ").strip()
            if not prompt: continue
            
            # Format prompt with the new training format (includes newlines)
            # This helps the model distinguish roles better.
            formatted_prompt = f"<user>\n{prompt}\n<assistant>\n"
            
            # Encode
            tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)
            input_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
            
            print("Assistant> ", end="", flush=True)
            
            generated_ids = input_ids
            # We reset cache_params per turn to allow fresh prompt processing, 
            # but KEEP the Dynamic Scratchpad (memory) for long-term recall.
            cache_params = None
            
            with torch.no_grad():
                for i in range(MAX_NEW_TOKENS):
                    # Temporary disable KV cache to bypass transformers' Mamba slow_forward bug
                    # We pass the full history (generated_ids) every time.
                    # This is slower but stable.
                    logits, _, memory, _ = model(
                        input_ids=generated_ids,
                        memory=memory,
                        use_cache=False
                    )
                    
                    # Selection logic
                    next_token_logits = logits[:, -1, :] / max(TEMPERATURE, 1e-6)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to tracking
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Decode and stream the character
                    char_token = next_token.item()
                    
                    if char_token == tokenizer.eos_token_id:
                        break
                    
                    # BytesTokenizer handles single bytes
                    # We convert back to character for streaming
                    char_out = tokenizer.decode([char_token])
                    print(char_out, end="", flush=True)
                    
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
