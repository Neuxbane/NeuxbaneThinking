import torch
import os
from model import NeuxbaneThinking, BPETokenizer

def test_inference():
    MODEL_PATH = "checkpoint/neuxbane_thinking_125m"
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer()
    model = NeuxbaneThinking()
    
    weights_path = MODEL_PATH + ".pth"
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device).to(torch.bfloat16 if "cuda" in str(device) else torch.float32)
    model.eval()
    
    prompt = "I need to find the derivative of f(x) = x^3 * ln(x)."
    # Exactly match train.py: text += f"<{role}>{content}"
    formatted_prompt = f"<user>{prompt}<assistant>"
    
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)
    input_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    print(f"Prompt: {formatted_prompt}")
    print("Generating...")
    
    # Use greedy for exact reproduction check
    generated_ids = input_ids
    memory = None
    cache_params = None
    
    with torch.no_grad():
        for i in range(100):
            logits, _, memory, _ = model(
                generated_ids,
                memory=memory,
                use_cache=False
            )
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            char = tokenizer.decode([next_token.item()])
            print(char, end="", flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    print("\nDone.")

if __name__ == "__main__":
    test_inference()
