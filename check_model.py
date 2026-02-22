import torch
from model import NeuxbaneThinking

def check():
    device = "cuda:1"
    model = NeuxbaneThinking()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.1f}M")
    
    # Try to move to GPU
    print(f"Moving to {device}...")
    model.to(torch.bfloat16).to(device)
    print("Model on device.")
    
    input_ids = torch.zeros((1, 128), dtype=torch.long, device=device)
    print(f"Input shape: {input_ids.shape}")
    print("Forward pass...")
    # model.train() # Test in train mode too
    logits, _, _, aux_loss = model(input_ids)
    print("Forward pass successful!")
    print(f"Logits shape: {logits.shape}")

    # Causal shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    print(f"Flat logits: {flat_logits.shape} | Flat labels: {flat_labels.shape}")
    
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(flat_logits, flat_labels)
    print(f"Loss successful: {loss.item()}")

if __name__ == "__main__":
    check()
