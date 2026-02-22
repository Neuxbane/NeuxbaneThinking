import os

# Set VRAM allocation configuration for better efficiency
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from model import NeuxbaneThinking, BPETokenizer
import signal
import sys

# Optional bitandbytes for memory efficiency
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    from torch.optim import AdamW

# Configuration
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoint/neuxbane_thinking_125m"
DATASETS_DIR = "datasets"
# High-performance hyper-parameters for convergence speed
LEARNING_RATE = 4e-4 
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1 
ACCUMULATION_STEPS = 16 
MAX_LENGTH = 64 # Small for stability
GRAD_CLIP = 1.0
WARMUP_STEPS = 50
USE_GRADIENT_CHECKPOINTING = False # Not needed for 125M

class JsonlDataset(IterableDataset):
    def __init__(self, directory, tokenizer, max_length):
        self.files = glob.glob(os.path.join(directory, "*.json*"))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        while True: # Infinite loop over files for infinite training
            for file_path in self.files:
                if not os.path.isfile(file_path): continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            
                            text = ""
                            if "conversations" in data:
                                # Use standard ChatML-like formatting with newlines for better breathing room
                                for msg in data["conversations"]:
                                    role = msg.get("role", "user")
                                    content = msg.get("content", "")
                                    if content: text += f"<{role}>\n{content}\n"
                            else:
                                query = data.get("query", data.get("instruction", ""))
                                response = data.get("response", data.get("output", ""))
                                if query: text += f"<user>\n{query}\n"
                                if response: text += f"<assistant>\n{response}\n"
                            
                            if not text.strip(): continue
                            
                            # Encode with label masking (only train on assistant part)
                            # Actually, for brevity let's just use the current simple training 
                            # but with BETTER formatting. This will already help.
                            encoded = self.tokenizer.encode(
                                text,
                                max_length=self.max_length,
                                add_special_tokens=True
                            )
                            # Convert to torch tensor
                            input_ids = torch.LongTensor(encoded)
                            
                            # Simple approach: train on everything (causal lm)
                            # If you want to mask user part, you'd need a more complex loop here.
                            yield input_ids, torch.ones_like(input_ids)
                        except Exception:
                            continue

def save_model(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 1. Save Base Mamba Model + Routers
    base_state = {k: v for k, v in model.state_dict().items() if not k.startswith("scratchpad_pool.")}
    torch.save(base_state, path + ".pth")
    
    # 2. Save Specialist Pool
    model.save_specialists()
    
    # 3. Save Optimizer & Config
    torch.save(optimizer.state_dict(), path + "_opt.pth")
    model.base_model.config.save_pretrained(os.path.dirname(path))
    print(f"\nCheckpoint saved: {path} (Mamba base + Specialist directory)")

def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # Initialize standard GPT-2 BPE Tokenizer
    tokenizer = BPETokenizer()

    # Initialize Model (GPT2-BPE-based 125M)
    model = NeuxbaneThinking()
    
    # Enable Gradient Checkpointing to save memory
    if USE_GRADIENT_CHECKPOINTING:
        print("Enabling Gradient Checkpointing for VRAM safety...")
        model.gradient_checkpointing_enable()
    
    # Load Model weights if exist
    base_weight_path = MODEL_PATH + ".pth"
    if os.path.exists(base_weight_path):
        print(f"Loading base Mamba weights from {base_weight_path}...")
        model.load_state_dict(torch.load(base_weight_path, map_location=device), strict=False)
        # Scratchpads are loaded automatically by NeuxbaneThinking.__init__ calling self.load_scratchpads()
    
    model.to(device)
    if device.type == "cuda":
        model.to(torch.bfloat16)
    else:
        model.to(torch.float32)
        
    model.train()

    # Prepare Optimizer (Use bitsandbytes Adam8bit if available to save ~6GB VRAM)
    if HAS_BNB:
        print("Using BitsAndBytes 8-bit AdamW optimizer for memory efficiency...")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        print("Using standard PyTorch AdamW optimizer...")
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing with Warmup for faster convergence
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        import math
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * (current_step - WARMUP_STEPS) / 10000)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Load Optimizer if exist
    if os.path.exists(MODEL_PATH + "_opt.pth"):
        print(f"Loading existing optimizer state from {MODEL_PATH}_opt.pth")
        optimizer.load_state_dict(torch.load(MODEL_PATH + "_opt.pth", map_location=device))
        
    criterion = nn.CrossEntropyLoss()

    dataset = JsonlDataset(DATASETS_DIR, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    print(f"Dataset files: {len(dataset.files)}")

    print("Starting infinite training... Press Ctrl+C to stop and save.")
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    
    accumulated_loss = 0
    step = 0
    try:
        print("Waiting for first batch...")
        for input_ids, attention_mask in dataloader:
            if step == 0: 
                print(f"First batch received! Input shape: {input_ids.shape}")
            input_ids = input_ids.to(device)
            
            # 1. FORWARD: Use Autocast + Mamba Seq Fallback
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids, 
                    memory=None,
                    use_cache=False
                )
                logits = outputs[0]
                aux_loss = outputs[-1]

                # Ensure logits is the actual tensor
                if isinstance(logits, tuple):
                    logits = logits[0]

                if step == 0:
                    print(f"Logits shape: {logits.shape}")

                # Causal shift
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Reshape for CrossEntropy
                # flat_logits should be [Batch * Seq, Vocab]
                # flat_labels should be [Batch * Seq]
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                
                if step == 0:
                    print(f"Flat logits: {flat_logits.shape} | Flat labels: {flat_labels.shape}")

                ce_loss = criterion(flat_logits, flat_labels)
                loss = (ce_loss + aux_loss) / ACCUMULATION_STEPS
            
            # 2. BACKWARD: Gradient Scaling (implicit in Autocast/Bfloat16 context)
            loss.backward()
            accumulated_loss += ce_loss.item() 

            # 3. OPTIMIZER STEP (Every N steps)
            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                print(f"Step: {step // ACCUMULATION_STEPS} | CE Loss: {accumulated_loss / ACCUMULATION_STEPS:.4f} | Aux: {aux_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                accumulated_loss = 0
                
                # Periodic cache clearing to avoid fragmentation
                if (step // ACCUMULATION_STEPS) % 100 == 0:
                    torch.cuda.empty_cache()
            
            step += 1
            
            # Backup save every 100 actual steps
            if step % 800 == 0:
                save_model(model, optimizer, MODEL_PATH)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")
        save_model(model, optimizer, MODEL_PATH)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        save_model(model, optimizer, MODEL_PATH)

if __name__ == "__main__":
    main()
