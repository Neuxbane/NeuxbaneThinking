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
    "dolci_think": -1,
    "tiny_think":-1,
    "medical-reasoning": 1000,
    "glaive-function-calling-v2-query": 1000,
    "ToolACE-query": 1000,
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
    # Print GPU memory usage before saving
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n[GPU Memory] Current Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB")

    print(f"Saving model to {path}...")
    # Strip torch.compile wrapper if present to save clean state_dict
    state_dict = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    torch.save(state_dict, path)

def main():
    # Setup
    data_dir = "datasets"
    tokenizer = ByteTokenizer()
    
    # Configuration
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384
    )

    # --- Loading Dataset FIRST (High CPU/Memory, No GPU) ---
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Loading data from {data_dir}...")
    dataset = JSONLDataset(data_dir, tokenizer, config.block_size, samplings=dataset_samplings)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # --- NOW claim GPU (Secure memory just before initializing model) ---
    device, device_name = set_device(min_memory_gb=4.0)
    
    model = Transformer(config).to(device)
    
    # Load existing model
    checkpoint_path = "model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path}")
        try:
            state = torch.load(checkpoint_path, map_location=device)
            # Remove _orig_mod. prefix if present from previous torch.compile saves
            if any(k.startswith("_orig_mod.") for k in state.keys()):
                print("Note: Striping '_orig_mod.' prefix from checkpoint keys.")
                state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
                
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    # Use torch.compile to speed up the model (especially with Flash Attention)
    try:
        print("Compiling model for performance...")
        # Keep original model for fallback if compilation results in OOM during execution
        uncompiled_model = model
        model = torch.compile(model)
        is_compiled = True
    except Exception as e:
        print(f"torch.compile failed early or not supported: {e}")
        uncompiled_model = model
        is_compiled = False

    # Signal handler for immediate save
    def signal_handler(sig, frame):
        # Save the raw model (uncompiled state_dict)
        save_model(uncompiled_model, checkpoint_path)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting training... Press Ctrl+C to interrupt and save.", flush=True)
    
    epoch = 0
    prev_avg_loss = None
    while True:
        model.train()
        total_loss = 0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            try:
                logits, loss, _, _ = model(x, y)
            except Exception as e:
                # Catch OOM during lazy compilation or first forward/backward pass
                if is_compiled and ("CUDA error: out of memory" in str(e) or "BackendCompilerFailed" in str(e)):
                    print(f"\n[!] torch.compile failed during execution: {e}")
                    print("Falling back to uncompiled model for stability.")
                    model = uncompiled_model
                    is_compiled = False
                    # Clear cache and retry step with original model
                    torch.cuda.empty_cache()
                    logits, loss, _, _ = model(x, y)
                else:
                    raise e
            
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

            # Periodically report GPU VRAM
            if step % 200 == 0:
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f" >>> [GPU VRAM] Allocated: {alloc:.2f}GB / Peak: {peak:.2f}GB", flush=True)
        
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
