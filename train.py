import os

# Set VRAM allocation configuration for better efficiency
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from model import NeuxbaneSSM, BPETokenizer
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
DEVICE = "cuda:0" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoint/neuxbane_thinking_125m"
DATASETS_DIR = "datasets"
# High-performance hyper-parameters for convergence speed
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1 
ACCUMULATION_STEPS = 8 
MAX_LENGTH = 512 # Increased for long reasoning chains
GRAD_CLIP = 1.0
WARMUP_STEPS = 20
USE_GRADIENT_CHECKPOINTING = True # Essential for high-density SSM on shared GPUs

class JsonlDataset(IterableDataset):
    def __init__(self, directory, tokenizer, max_length):
        self.files = glob.glob(os.path.join(directory, "*.json*"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for file_path in self.files:
            if not os.path.isfile(file_path): continue
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "conversations" in data:
                            msgs = data["conversations"]
                        else:
                            q = data.get("query", data.get("instruction", ""))
                            r = data.get("response", data.get("output", ""))
                            msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
                        self.samples.append(msgs)
                    except: continue

    def __iter__(self):
        import random
        while True:
            shuffled = list(self.samples)
            random.shuffle(shuffled)
            for msgs in shuffled:
                input_ids = []
                labels = []
                
                for msg in msgs:
                    role = msg["role"]
                    content = msg["content"]
                    
                    # Add role token
                    role_token = self.tokenizer.user_token_id if role == "user" else self.tokenizer.assistant_token_id
                    input_ids.append(role_token)
                    labels.append(-100) # Don't predict role tags
                    
                    # Add newline
                    nl_token = self.tokenizer.tokenizer.encode("\n", add_special_tokens=False)[0]
                    input_ids.append(nl_token)
                    labels.append(-100)
                    
                    # Add content
                    tokens = self.tokenizer.tokenizer.encode(content, add_special_tokens=False)
                    input_ids.extend(tokens)
                    
                    if role == "user":
                        labels.extend([-100] * len(tokens))
                    else:
                        labels.extend(tokens)
                        
                    # Add terminal newline
                    input_ids.append(nl_token)
                    if role == "user":
                        labels.append(-100)
                    else:
                        labels.append(nl_token)

                # Truncate
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                
                yield torch.LongTensor(input_ids), torch.LongTensor(labels)

def save_model(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 1. Save all weights (Simplified since we built our own model)
    torch.save(model.state_dict(), path + ".pth")
    
    # 2. Save Specialist Grid specifically if needed separate
    # model.save_specialists()
    
    # 3. Save Optimizer state
    torch.save(optimizer.state_dict(), path + "_opt.pth")
    print(f"\nCheckpoint saved: {path}.pth")

def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # Initialize standard GPT-2 BPE Tokenizer
    tokenizer = BPETokenizer()

    # Initialize Model (GPT2-BPE-based 125M)
    model = NeuxbaneSSM()
    
    # Enable Gradient Checkpointing to save memory
    if USE_GRADIENT_CHECKPOINTING:
        print("Enabling Gradient Checkpointing for VRAM safety...")
        model.gradient_checkpointing_enable()
    
    # Load Model weights if exist
    base_weight_path = MODEL_PATH + ".pth"
    if os.path.exists(base_weight_path):
        print(f"Loading base Mamba weights from {base_weight_path}...")
        model.load_state_dict(torch.load(base_weight_path, map_location=device), strict=False)
        # Scratchpads are loaded automatically by NeuxbaneSSM.__init__ calling self.load_scratchpads()
    
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
        
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

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
        for input_ids, labels in dataloader:
            if step == 0: 
                print(f"First batch received! Input shape: {input_ids.shape}")
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
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
                shift_labels = labels[..., 1:].contiguous()
                
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
                
                # Normalize printed loss by batch/seq for readability
                avg_ce = (accumulated_loss / ACCUMULATION_STEPS)
                aux_val = aux_loss.item() if hasattr(aux_loss, "item") else aux_loss
                print(f"Step: {step // ACCUMULATION_STEPS} | Per-Token Loss: {avg_ce:.4f} | Aux: {aux_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
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
