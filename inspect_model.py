import torch
from model import Transformer, TransformerConfig, ByteTokenizer

def inspect_model():
    # Use the same configuration as in train.py
    tokenizer = ByteTokenizer()
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384,
        n_scratchpad=64
    )
    
    # Initialize model without loading existing weights
    model = Transformer(config)
    
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*50)
    print(f"Vocab Size:      {config.vocab_size}")
    print(f"Block Size:      {config.block_size}")
    print(f"Layers:          {config.n_layer}")
    print(f"Heads:           {config.n_head}")
    print(f"Embedding Dim:   {config.n_embd}")
    print(f"Scratchpad Size: {config.n_scratchpad}")
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
    dummy_input = torch.randint(0, config.vocab_size, (1, 10))
    logits, loss, scratchpad, _ = model(dummy_input)
    
    print("\nForward Pass Verification:")
    print(f"  Input Shape:      {dummy_input.shape}")
    print(f"  Logits Shape:     {logits.shape} (Expected: [1, 10, {config.vocab_size}])")
    if scratchpad is not None:
        print(f"  Scratchpad Shape: {scratchpad.shape} (Expected: [1, {config.n_scratchpad}, {config.n_embd}])")
    
    print("="*50)

if __name__ == "__main__":
    inspect_model()
