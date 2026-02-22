import torch
from model import NeuxbaneThinking, BPETokenizer

def test_generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer()
    model = NeuxbaneThinking()
    model.to(device).to(torch.bfloat16)
    model.eval()
    
    text = "Query: What is 2+2?\nAssistant: <think>"
    tokens = tokenizer.encode(text)
    input_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    print(f"Generating on {device}...")
    try:
        # returns token ids
        output_ids = model.generate(input_ids, max_new_tokens=10)
        decoded = tokenizer.decode(output_ids[0].tolist())
        print("Generation successful!")
        print(f"Output: {decoded}")
    except Exception as e:
        print(f"Generation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generate()
