import sys
import os

# Add root to sys.path to allow imports if run from training/ directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import Transformer, TransformerConfig, ByteTokenizer
from gpu_utils import set_device

# --- REWARD FUNCTIONS ---

def extract_answer(text):
    """
    Tries to extract a numerical answer from text.
    First looks for bolded text (**12**), then for the last number if nothing else found.
    """
    # Try finding something inside double asterisks
    bold_match = re.search(r'\*\*(.*?)\*\*', text)
    if bold_match:
        answer = bold_match.group(1).strip()
        # Remove common math symbols for comparison
        answer = re.sub(r'[\$°%cm\^2\s]', '', answer)
        return answer
    
    # Otherwise look for the last number/fraction in the text
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:/\d+)?', text)
    if numbers:
        return numbers[-1]
    
    return None

def reward_math(problem, completion, gt_answer):
    """
    Reward function for math problems.
    - Correct answer: +1.0
    - Incorrect answer: 0.0
    - Format penalty (no <think> tags): -0.1
    """
    reward = 0.0
    
    # Check if answer is correct
    model_answer = extract_answer(completion)
    
    if model_answer:
        # Simple string comparison after some normalization
        if model_answer.strip().lower() == str(gt_answer).strip().lower():
            reward += 1.0
    
    # Check formatting
    if not ("<think>" in completion and "</think>" in completion):
        reward -= 0.1
    
    return reward

# --- MATH GENERATOR ---
import random

def generate_math_problem():
    """
    Generates a random math problem, returns (prompt, ground_truth, answer).
    """
    problem_type = random.choice(["addition", "subtraction", "multiplication", "division", "percentage", "algebra"])
    
    if problem_type == "addition":
        a, b = random.randint(1, 100), random.randint(1, 100)
        prompt = f"What is {a} + {b}?"
        answer = str(a + b)
        gt = f"The sum is **{answer}**."
    elif problem_type == "subtraction":
        a, b = random.randint(1, 100), random.randint(1, 100)
        if a < b: a, b = b, a
        prompt = f"What is {a} - {b}?"
        answer = str(a - b)
        gt = f"The difference is **{answer}**."
    elif problem_type == "multiplication":
        a, b = random.randint(1, 20), random.randint(1, 20)
        prompt = f"What is the product of {a} and {b}?"
        answer = str(a * b)
        gt = f"The product is **{answer}**."
    elif problem_type == "division":
        b = random.randint(1, 12)
        ans = random.randint(1, 12)
        a = b * ans
        prompt = f"What is {a} divided by {b}?"
        answer = str(ans)
        gt = f"The result is **{answer}**."
    elif problem_type == "percentage":
        total = random.choice([20, 40, 50, 60, 80, 100, 200, 400, 500])
        p = random.choice([5, 10, 15, 20, 25, 50, 75])
        ans_val = (p * total) // 100
        prompt = f"What is {p}% of {total}?"
        answer = str(ans_val)
        gt = f"{p}% of {total} is **{answer}**."
    elif problem_type == "algebra":
        a = random.randint(1, 20)
        ans = random.randint(1, 30)
        b = a + ans
        prompt = f"Solve for x: x + {a} = {b}"
        answer = str(ans)
        gt = f"The value of x is **{answer}**."

    full_prompt = f"<bos><role>user</role>{prompt}<role>assistant</role>"
    return full_prompt, gt, answer

# --- Dataset and Training ---

class MathTemplateDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prompt, gt, answer = generate_math_problem()
        return prompt, gt, answer


def get_log_probs(model, input_ids, gen_ids, eos_id):
    """
    Calculates the log-probs of the generated gen_ids tokens.
    input_ids contains [prompt_ids + gen_ids].
    """
    logits, _, _, _ = model(input_ids) # (B, S, V)
    prompt_len = input_ids.size(1) - gen_ids.size(1)
    
    # Get logits relevant to the output tokens
    # logits[i] predicts input_ids[i+1]
    # We want logits for input_ids[prompt_len] to [L-1]
    relevant_logits = logits[:, prompt_len-1:-1, :] 
    log_probs = F.log_softmax(relevant_logits, dim=-1)
    
    # Gather the log-probs of the actual tokens sampled
    token_log_probs = torch.gather(log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
    
    # Mask out log-probs after the first <eos>
    # Find the position of the first <eos> for each sequence in the batch
    # We create a mask: 1 before and including first <eos>, 0 after
    eos_positions = (gen_ids == eos_id).long().argmax(dim=1)
    # If <eos> not found, argmax returns 0. Check if it's really there.
    has_eos = (gen_ids == eos_id).any(dim=1)
    
    # Mask: (B, seq_len)
    mask = torch.ones_like(gen_ids, dtype=torch.float)
    for i in range(gen_ids.size(0)):
        if has_eos[i]:
            mask[i, eos_positions[i]+1:] = 0.0
            
    return token_log_probs * mask

def train():
    # Setup
    device, _ = set_device(min_memory_gb=1.0)
    tokenizer = ByteTokenizer()
    eos_id = tokenizer.special_to_id["<eos>"]
    
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384
    )
    
    model = Transformer(config).to(device)
    
    # Resolve paths relative to root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_path = os.path.join(root_dir, "model.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Starting from scratch...")

    # RL Hyperparameters
    num_generations = 4  # G in GRPO
    lr = 1e-6 # Much smaller for RL
    max_new_tokens = 256
    batch_size = 2 # Number of prompts per batch
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    dataset = MathTemplateDataset(size=5000) # Infinite-like dataset
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    print(f"Starting RL training with generated math problems...")
    
    for epoch in range(10):
        for step, (prompts, gts, answers) in enumerate(dataloader):
            batch_loss = 0
            all_rewards = []
            
            # For each prompt in batch, we sample G completions
            for prompt, gt, answer in zip(prompts, gts, answers):
                prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                
                # Sample G completions
                # To do this efficiently, we repeat prompt_ids G times
                batch_prompt_ids = prompt_ids.repeat(num_generations, 1)
                
                # Generate
                # model.generate returns (B, prompt_len + max_new_tokens)
                with torch.no_grad():
                    # We want to use different samples for each of the G completions
                    # but model.generate with torch.no_grad is what we have.
                    # It samples internally using torch.multinomial.
                    full_ids = model.generate(batch_prompt_ids, max_new_tokens, temperature=0.8)
                
                # Extract the generated portion
                gen_ids = full_ids[:, prompt_ids.size(1):]
                
                # Calculate rewards for each generation
                rewards = []
                completions = []
                for i in range(num_generations):
                    completion = tokenizer.decode(gen_ids[i].tolist())
                    completions.append(completion)
                    r = reward_math(prompt, completion, answer)
                    rewards.append(r)
                
                all_rewards.append(sum(rewards)/len(rewards))
                
                # GRPO Advantage Calculation: Standardize rewards across the group
                rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
                mean_r = rewards_tensor.mean()
                std_r = rewards_tensor.std() + 1e-8
                advantages = (rewards_tensor - mean_r) / std_r
                
                # Policy Gradient Loss
                # We need to re-run the model to get gradients for gen_ids
                # We use the full sequence but mask out the prompt part
                log_p = get_log_probs(model, full_ids, gen_ids, eos_id)
                
                # Loss = - advantage * log_prob (summed over tokens, averaged over group)
                # We also need to avoid updating if the sequence hit EOS early, 
                # but for simplicity we'll just use the full generation window.
                # Mask out tokens after <eos> if we want to be more precise.
                
                # Simple policy gradient loss:
                # l_prob = sum of log-probs for the sequence
                seq_log_p = log_p.sum(dim=1)
                loss = -(advantages * seq_log_p).mean()
                
                batch_loss += loss / batch_size
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                avg_r = sum(all_rewards)/len(all_rewards)
                print(f"Epoch {epoch} | Step {step} | Loss {batch_loss.item():.4f} | Avg Reward {avg_r:.4f}")
                
                # Print one sample
                print(f"Sample Completion for '{prompts[0][:50]}...':")
                print(f"Output: {completions[0][:150]}...")
                print(f"Reward: {rewards[0]}")

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    train()
