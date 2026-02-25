import torch
import torch.nn as nn
from model import Transformer, TransformerConfig, ByteTokenizer
import time
import numpy as np

def generate_random_batch(batch_size, seq_len, vocab_size, device):
    """Generates a random batch of input tokens and targets."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Mask some targets to simulate 'ignore_index' like in real training
    mask = torch.rand((batch_size, seq_len), device=device) < 0.2
    targets[mask] = -1
    return input_ids, targets

def analyze_speed(model, config, device):
    batch_size = 2 # Reduced from 4 to avoid OOM in tight environments
    seq_len = config.block_size
    
    print("\n" + "="*50)
    print(f"SPEED ANALYSIS (Device: {device})")
    print("="*50)
    
    # --- Training Speed (Forward + Backward) ---
    input_ids, targets = generate_random_batch(batch_size, seq_len, config.vocab_size, device)
    
    # Warmup
    for _ in range(3):
        logits, loss, _, _ = model(input_ids, targets=targets)
        loss.backward()
        model.zero_grad()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    num_iters = 10
    total_tokens = num_iters * batch_size * seq_len
    
    for _ in range(num_iters):
        logits, loss, _, _ = model(input_ids, targets=targets)
        loss.backward()
        model.zero_grad()
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_train_time = (end_time - start_time) / num_iters
    train_tokens_per_sec = total_tokens / (end_time - start_time)
    
    print(f"Training (Fwd+Bwd):")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Seq Len:         {seq_len}")
    print(f"  Avg Iter Time:   {avg_train_time:.4f}s")
    print(f"  Throughput:      {train_tokens_per_sec:.2f} tokens/s")
    
    # --- Inference Speed (Auto-regressive Generation) ---
    gen_len = 50
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    
    # Warmup
    for _ in range(2):
        _ = model.generate(prompt, max_new_tokens=5)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    gen_iters = 5
    for _ in range(gen_iters):
        _ = model.generate(prompt, max_new_tokens=gen_len)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_gen_time = (end_time - start_time) / gen_iters
    inference_tokens_per_sec = (gen_iters * gen_len) / (end_time - start_time)
    
    print(f"\nInference (Generation):")
    print(f"  New Tokens:      {gen_len}")
    print(f"  Avg Gen Time:    {avg_gen_time:.4f}s")
    print(f"  Throughput:      {inference_tokens_per_sec:.2f} tokens/s")
    print("-" * 50)

def inspect_model():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the same configuration as in train.py
    tokenizer = ByteTokenizer()
    # Updated configuration for Multi-Page Routed Scratchpad
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_kv_head=6,
        n_embd=384,
        n_scratchpad=512,    # Capacity per scratchpad
        num_scratchpads=4    # Number of scratchpad pages
    )
    
    # Initialize model
    model = Transformer(config).to(device)
    
    # Load existing weights if they exist
    import os
    if os.path.exists("model.pth"):
        print("Loading weights from model.pth...")
        try:
            state_dict = torch.load("model.pth", map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print("Successfully loaded model.pth")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    else:
        print("No model.pth found. Using randomly initialized weights.")

    model.train() # For training speed analysis
    
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*50)
    print(f"Vocab Size:      {config.vocab_size}")
    print(f"Block Size:      {config.block_size}")
    print(f"Layers:          {config.n_layer}")
    print(f"Heads:           {config.n_head}")
    print(f"KV Heads:        {config.n_kv_head}")
    print(f"Embedding Dim:   {config.n_embd}")
    print(f"Scratchpad Size: {config.n_scratchpad}")
    print("-" * 50)
    
    reported_params = model.get_num_params()
    print(f"Weight-Tied Params: {reported_params:,}")
    print("-" * 50)
    
    # --- Sanity Check: Loss and Gradients ---
    expected_loss = np.log(config.vocab_size)
    dummy_input, dummy_targets = generate_random_batch(4, 128, config.vocab_size, device)
    
    # Explicitly ensure grad is enabled
    torch.set_grad_enabled(True)
    model.zero_grad()
    
    logits, loss, scratchpad, _ = model(dummy_input, targets=dummy_targets)
    
    print("\nSanity Checks:")
    print(f"  Logits Shape:     {logits.shape}")
    print(f"  Logits NaN/Inf:   {torch.isnan(logits).any().item()}/{torch.isinf(logits).any().item()}")
    print(f"  Current Loss:     {loss.item():.4f} (Random Base: ~{expected_loss:.2f})")
    
    # Check if gradients flow to all parts
    loss.backward()
    
    trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params_with_grad = [name for name, p in trainable_params if p.grad is not None]
    params_without_grad = [name for name, p in trainable_params if p.grad is None]
    
    has_grad = len(params_without_grad) == 0
    if not has_grad:
        print(f"  [MISSING GRADIENTS] {len(params_without_grad)} items: {params_without_grad[:5]}...")
        print("  NOTE: Missing gradients in the last layer's 'write' heads are expected,")
        print("  as the final scratchpad update is never read by any subsequent token layer.")
    else:
        print(f"  Gradients flow:   {has_grad} ({len(params_with_grad)}/{len(trainable_params)} modules)")

    # Scratchpad verification
    if scratchpad is not None:
        print(f"  Scratchpad Shape: {scratchpad.shape}")
        print(f"  Scratchpad NaN:   {torch.isnan(scratchpad).any().item()}")

    # --- Sample Training Data Check ---
    print("\nTraining Batch Sample (first item):")
    tokens = dummy_input[0].tolist()
    labels = dummy_targets[0].tolist()
    
    # Decode first 20 tokens as a sample
    decoded_inputs = tokenizer.decode(tokens[:20])
    valid_labels = [l for l in labels[:20] if l != -1]
    decoded_labels = tokenizer.decode(valid_labels) if valid_labels else "None"
    
    print(f"  Input tokens (ids): {tokens[:10]}...")
    print(f"  Input decoded:      {repr(decoded_inputs)}")
    print(f"  Targets (first 5):  {labels[:5]}")
    
    # --- Speed Analysis ---
    analyze_speed(model, config, device)

    print("\nSummary Status:")
    is_good = True
    if torch.isnan(logits).any():
        print("  [CRITICAL] NaNs detected in model outputs.")
        is_good = False
    if not has_grad:
        print(f"  [ERROR] {len(params_without_grad)} parameters are disconnected from the loss.")
        is_good = False
    
    if is_good:
        print("  [SUCCESS] Model and script checks passed.")
    else:
        print("  [FAILURE] Issues detected. Check gradient flow or configuration.")

    print("="*50)


if __name__ == "__main__":
    inspect_model()

