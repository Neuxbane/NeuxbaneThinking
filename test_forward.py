import torch
from model import NeuxbaneThinking, BPETokenizer

def test_model():
    device = "cpu" # Force CPU for testing logic as GPU is full
    tokenizer = BPETokenizer()
    model = NeuxbaneThinking()
    model.to(device).to(torch.float32)
    model.eval()
    
    text = "Hello world"
    tokens = tokenizer.encode(text)
    input_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    # Force base model and all modules to float32
    model = model.to(torch.float32)

    print(f"Testing forward pass on {device}...")
    try:
        # returns logits, cache_params, memory
        logits, _, _ = model(input_ids, use_cache=False)
        print("Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
