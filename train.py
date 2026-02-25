import os
import json
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Transformer, TransformerConfig, ByteTokenizer
from gpu_utils import set_device
import signal
import sys
import random

dataset_samplings = {
    "claude-4.5-high-reasoning": -1,
    "stem_reasoning": 0,
}

# --- Dataset ---
class JSONLDataset(Dataset):
    def __init__(self, data_root, tokenizer, block_size, samplings=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        
        if samplings is None:
            samplings = {}

        # Search specifically for the files defined in samplings
        potential_dirs = [data_root, "dmp"]
        all_entries = []
        
        for name, limit in samplings.items():
            found_file = None
            for d in potential_dirs:
                if os.path.exists(d):
                    # Check for direct file or recursive glob
                    candidate = os.path.join(d, f"{name}.jsonl")
                    if os.path.exists(candidate):
                        found_file = candidate
                        break
                    matches = glob.glob(os.path.join(d, f"**/{name}.jsonl"), recursive=True)
                    if matches:
                        found_file = matches[0]
                        break
            
            if not found_file:
                print(f"Warning: Dataset file for '{name}' not found.")
                continue

            print(f"Loading '{name}' from {found_file} (limit: {limit})...")
            
            reservoir = []
            with open(found_file, 'r', encoding='utf-8') as f:
                if limit == -1:
                    # Load all
                    for line in f:
                        try:
                            reservoir.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                else:
                    # Memory-efficient Reservoir Sampling
                    for idx, line in enumerate(f):
                        try:
                            if idx < limit:
                                reservoir.append(json.loads(line))
                            else:
                                j = random.randint(0, idx)
                                if j < limit:
                                    reservoir[j] = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue
            
            all_entries.extend(reservoir)
            print(f"Added {len(reservoir)} items from '{name}'.")
        
        # Final mix of the sampled entries from different files
        random.shuffle(all_entries)
        print(f"Total mixed items for training: {len(all_entries)}")

        for data in all_entries:
            convs = data.get("conversations", [])
            tools = data.get("tools")
            
            tokens = []
            targets = []
            
            # Start with BOS and tools definition (masked)
            header_text = f"<bos>" + (f"<tools>{json.dumps(tools)}</tools>" if tools else "")
            header_toks = self.tokenizer.encode(header_text)
            tokens.extend(header_toks)
            targets.extend([-1] * len(header_toks))
            
            for i, conv in enumerate(convs):
                role = conv["role"]
                content = conv.get("content", "")
                
                # Role and internal tags (masked)
                role_tags = f"<role>{role}</role>"
                role_toks = self.tokenizer.encode(role_tags)
                
                # Content
                if role == "assistant" and "tool_calls" in conv:
                    content = f"<tool_calls>{json.dumps(conv['tool_calls'])}</tool_calls>{content}"
                content_toks = self.tokenizer.encode(content)
                
                # Assistant turns have content + eos as targets
                # User turns are masked
                tokens.extend(role_toks)
                targets.extend([-1] * len(role_toks))
                
                tokens.extend(content_toks)
                if role == "assistant":
                    targets.extend(content_toks)
                else:
                    targets.extend([-1] * len(content_toks))
                
                if role == "assistant":
                    eos_toks = self.tokenizer.encode("<eos>")
                    tokens.extend(eos_toks)
                    targets.extend(eos_toks)
            
            # Chunking correctly with paired labels
            for j in range(0, len(tokens), self.block_size):
                chunk_tokens = tokens[j : j + self.block_size + 1]
                chunk_targets = targets[j : j + self.block_size + 1]
                if len(chunk_tokens) > 1:
                    # We store both so targets align with inputs
                    self.examples.append((chunk_tokens, chunk_targets))
        
        print(f"Successfully processed {len(self.examples)} total chunks for training.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, targets = self.examples[idx]
        
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)

        # Shifted targets: input tokens[i] predicts target targets[i+1]
        length = min(len(tokens) - 1, self.block_size)
        
        x[:length] = torch.tensor(tokens[:length], dtype=torch.long)
        y[:length] = torch.tensor(targets[1:length+1], dtype=torch.long)
        
        return x, y

# --- Main Training Logic ---

def save_model(model, path="model.pth"):
    print(f"\nSaving model to {path}...")
    torch.save(model.state_dict(), path)

def main():
    # Setup
    # NOTE: Delay GPU selection until AFTER data is loaded to avoid reserving GPU
    # while performing potentially heavy CPU-side dataset loading. Create model
    # on CPU, load data, then allocate GPU and move model there.
    data_dir = "datasets"

    tokenizer = ByteTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384
    )
    
    # Create model on CPU first to avoid early GPU allocations
    cpu_device = torch.device('cpu')
    model = Transformer(config).to(cpu_device)
    
    # Load existing model (into CPU first)
    checkpoint_path = "model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path}")
        try:
            state = torch.load(checkpoint_path, map_location=cpu_device)
            try:
                model.load_state_dict(state)
                print("Checkpoint loaded (strict match). Continuing training from checkpoint.")
            except RuntimeError as e_strict:
                print(f"\n[!] Strict load failed: {e_strict}")
                print("Attempting non-strict load (will ignore unexpected/missing keys)...")
                try:
                    model.load_state_dict(state, strict=False)
                    print("Loaded checkpoint with non-strict matching. Some parameters/buffers may be left at initialization values.")
                except Exception as e_nonstrict:
                    print(f"Non-strict load also failed: {e_nonstrict}")
                    print("Proceeding to train with newly initialized model (checkpoint not loaded).")
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            print("Proceeding to train with newly initialized model.")

    dataset = JSONLDataset(data_dir, tokenizer, config.block_size, samplings=dataset_samplings)

    # Now that data is loaded, select the device and move model there. This
    # prevents holding GPU resources while parsing/loading datasets.
    device, device_name = set_device(min_memory_gb=4.0)
    print(f"Using device: {device_name}")

    # Move model to chosen device
    model = model.to(device)

    # Use pin_memory for faster host->device transfers when using CUDA
    pin_memory = True if device.type == 'cuda' else False
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=pin_memory)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Signal handler for immediate save
    def signal_handler(sig, frame):
        save_model(model, checkpoint_path)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting training... Press Ctrl+C to interrupt and save.", flush=True)
    
    epoch = 0
    prev_avg_loss = None
    while True:
        model.train()
        total_loss = 0
        for step, (x, y) in enumerate(dataloader):
            # Use non_blocking transfer when pin_memory is enabled
            x = x.to(device, non_blocking=True) if pin_memory else x.to(device)
            y = y.to(device, non_blocking=True) if pin_memory else y.to(device)
            
            logits, loss, _, _ = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            running_avg = total_loss / (step + 1)
            
            delta_str = ""
            if prev_avg_loss is not None:
                delta = running_avg - prev_avg_loss
                delta_str = f" | Delta: {delta:+.4f}"

            print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {running_avg:.4f}{delta_str}", flush=True)
        
        avg_loss = total_loss / len(dataloader)
        prev_avg_loss = avg_loss
        
        epoch += 1
        
        # Auto-save every 50 epochs just in case
        if epoch % 50 == 0:
            save_model(model, checkpoint_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # This is also caught by signal_handler but just to be sure
        pass
