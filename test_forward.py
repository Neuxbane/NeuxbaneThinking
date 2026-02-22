import torch
from model import NeuxbaneThinking, BPETokenizer

def test_model():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer()
    model = NeuxbaneThinking()
    model.to(device).to(torch.bfloat16 if "cuda" in str(device) else torch.float32)
    model.eval()
    
    text = "Hello world"
    tokens = tokenizer.encode(text)
    input_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    # model = model.to(torch.bfloat16) # handled above

    print(f"Testing forward pass on {device}...")
    try:
        # returns logits, cache_params, memory, total_aux_loss
        logits, _, _, aux_loss = model(input_ids, use_cache=False)
        print("Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"Aux Loss: {aux_loss:.4f}")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
