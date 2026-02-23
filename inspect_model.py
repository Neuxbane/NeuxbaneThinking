import torch
from model import Transformer, TransformerConfig, ByteTokenizer
from gpu_utils import set_device

def inspect_model():
    # Set the device using the utility script
    device, device_name = set_device(min_memory_gb=1.0)
    
    # Use the same configuration as in train.py
    tokenizer = ByteTokenizer()
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=1024,
        n_layer=6,
        n_head=6,
        n_kv_head=2,
        n_embd=384,
        n_scratchpad=64
    )
    
    # Initialize model without loading existing weights
    model = Transformer(config).to(device)
    
    # Reset GPU stats if using CUDA
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    print("\n" + "="*50)
    print(f"MODEL ARCHITECTURE INSPECTION (Device: {device_name})")
    print("="*50)
    print(f"Vocab Size:      {config.vocab_size}")
    print(f"Block Size:      {config.block_size}")
    print(f"Layers:          {config.n_layer}")
    print(f"Heads:           {config.n_head}")
    print(f"KV Heads:        {config.n_kv_head}")
    print(f"Embedding Dim:   {config.n_embd}")
    print(f"Scratchpad Slots:{config.n_scratchpad}")
    print(f"Scratchpad Dim:  {config.n_embd}")
    print(f"Total Sp Params: {config.n_scratchpad * config.n_embd}")
    print("-" * 50)
    
    # Calculate parameters manualy to verify get_num_params()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Note: get_num_params() in model.py subtracts the tied weights
    reported_params = model.get_num_params()
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Weight-Tied Params:   {reported_params:,} (reported by model)")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * 50)
    
    print("\nModule Breakdown:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name: <15} | {num_params:,} parameters")
        if name == 'transformer':
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                print(f"    {sub_name: <13} | {sub_params:,} parameters")

    # Run a dummy forward pass to verify shapes
    dummy_input = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    logits, loss, _ = model(dummy_input)
    
    print("\nForward Pass Verification:")
    print(f"  Input Shape:      {dummy_input.shape}")
    print(f"  Logits Shape:     {logits.shape} (Expected: [1, 74, {config.vocab_size}])")
    
    # Run a generation test to verify KV caching and iterative forward
    print("\nGeneration Verification (5 tokens):")
    gen_tokens = 5
    generated_idx = model.generate(dummy_input, max_new_tokens=gen_tokens)
    
    print(f"  Input Tokens:     {dummy_input.size(1)}")
    print(f"  Generated Shape:  {generated_idx.shape} (Expected: [1, {10 + gen_tokens}])")

    # Sample Test Train (Verification of Backprop)
    print("\nSample Training Step (No Save):")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    # 512 is standard block_size, let's test a full chunk
    train_input = torch.randint(0, config.vocab_size, (2, 512)).to(device)
    train_targets = torch.randint(0, config.vocab_size, (2, 512)).to(device)
    
    # Clear memory peak again before actual intensive train test
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    logits, loss, _ = model(train_input, targets=train_targets)
    print(f"  Forward Pass Loss: {loss.item():.4f}")
    
    loss.backward()
    print("  Backward Pass:    [OK] Gradients computed.")
    
    # Check one specific gradient to be sure
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  Gradient Norm:    {grad_norm:.4f}")
    
    optimizer.step()
    print("  Optimizer Step:   [OK] Weights updated.")
    
    # Report GPU memory peak
    if device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n[GPU Memory] Peak: {peak_vram:.2f} GB")
    
    # Try decoding a small segment
    decoded = tokenizer.decode(generated_idx[0].tolist())
    print(f"  Decoded Sample:   {repr(decoded[:50])}...")

    print("="*50)

if __name__ == "__main__":
    inspect_model()
