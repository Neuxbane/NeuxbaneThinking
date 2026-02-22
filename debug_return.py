import torch
from model import NeuxbaneThinking, BytesTokenizer

def test():
    tokenizer = BytesTokenizer()
    model = NeuxbaneThinking()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    input_ids = torch.randint(0, 255, (1, 10)).to(device)
    
    print("Calling model...")
    out = model(input_ids=input_ids, use_cache=False)
    print(f"Return type: {type(out)}")
    if out is None:
        print("Return is None!")
    else:
        print(f"Return values lengths: {len(out)}")
        try:
            logits, cache, memory = out
            print("Unpacked successfully")
        except Exception as e:
            print(f"Unpacking failed: {e}")

if __name__ == "__main__":
    test()
