import torch
import torch.nn as nn
import torch.utils.checkpoint
import os
from transformers import MambaForCausalLM, MambaConfig, AutoTokenizer
from transformers.models.mamba.modeling_mamba import MambaCache
from typing import Optional, Tuple, Union, List, Any

class BPETokenizer:
    def __init__(self, model_id: str = "openai-community/gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.eos_token_id
        self.think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
        self.end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.vocab_size = len(self.tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_length)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

class DynamicScratchpad(nn.Module):
    def __init__(self, hidden_size: int, num_pads: int = 256, rank: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pads = num_pads
        self.rank = rank
        
        # Low-rank bottleneck to keep 72 specialists under 125M-130M budget
        self.in_proj = nn.Linear(hidden_size, rank * 3, bias=False) # Q, U, F
        self.out_proj = nn.Linear(rank, hidden_size, bias=False)
        
        self.gate = nn.Parameter(torch.zeros(1))
        # Lightweight evolution
        self.evolve = nn.Sequential(
            nn.Linear(hidden_size, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, hidden_size, bias=False)
        )
        self.evolve_gate = nn.Parameter(torch.zeros(1))
        self.diffusion_kernel = nn.Parameter(torch.eye(num_pads))
        self.mem_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, (seq_len), hidden_size]
        # memory: [batch, num_pads, hidden_size]
        
        # Consistent 3D handling
        was_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            was_2d = True
            
        batch, seq, _ = x.shape
        x_dtype = x.dtype
        
        # Projections via bottleneck
        projs = self.in_proj(x)
        q, u, f = torch.split(projs, self.rank, dim=-1)
        
        # Project memory to rank space for retrieval and update
        memory_rank = memory @ self.in_proj.weight[:self.rank].T # [B, Pads, Rank]

        # Attention over memory pads
        sim = torch.matmul(q, memory_rank.transpose(-1, -2)) # [B, S, Pads]
        attn = torch.softmax(sim / (self.rank**0.5), dim=-1)
        
        # Retrieval
        retrieved_rank = torch.matmul(attn, memory_rank) # [B, S, Rank]
        retrieved = self.out_proj(retrieved_rank)
        x_out = x + self.gate * retrieved
        
        # Update Logic
        f_gate = torch.sigmoid(f)
        
        # Specialists interaction weight
        if weight is not None:
             u = u * weight
             f_gate = f_gate * weight
             
        # Causal scan for memory is too slow during training, so we use a weighted-average update
        # for tokens in the same sequence.
        # attn: [B, S, Pads], u: [B, S, Rank]
        # We want u_pads: [B, Pads, Rank]
        u_pads = torch.matmul(attn.transpose(-1, -2), u)
        f_pads = torch.matmul(attn.transpose(-1, -2), f_gate)
        
        # Update memory in rank space
        memory_rank = memory_rank * (1.0 - f_pads.clamp(0, 1)) + u_pads
        memory = self.out_proj(memory_rank)
        
        # Evolution & Diffusion
        memory = torch.matmul(self.diffusion_kernel, memory)
        memory = memory + self.evolve_gate * self.evolve(memory)
        memory = self.mem_norm(memory)
        
        if was_2d:
            x_out = x_out.squeeze(1)
            
        return x_out, memory

class NeuxbaneThinking(nn.Module):
    def __init__(self, model_id_or_path: Optional[str] = None, checkpoint_dir: str = "checkpoint"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        if model_id_or_path:
            self.base_model = MambaForCausalLM.from_pretrained(model_id_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            config = MambaConfig(vocab_size=50260, hidden_size=768, state_size=16, num_hidden_layers=9, expand=2, conv_kernel=4, use_cache=True, rms_norm_eps=1e-5, torch_dtype=torch.bfloat16)
            self.base_model = MambaForCausalLM(config)
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_ropes = 8
        
        # Specialist Grid: 9 Layers x 8 Ropes
        self.specialist_grid = nn.ModuleList([
            nn.ModuleDict({
                f"rope_{j}": DynamicScratchpad(self.hidden_size) for j in range(self.num_ropes)
            }) for i in range(self.num_layers)
        ])
        
        # Routers: One per layer deciding which "rope" to pull
        self.routers = nn.ModuleList([nn.Linear(self.hidden_size, self.num_ropes) for _ in range(self.num_layers)])
        
        # Per-rope initial memory
        self.rope_init = nn.Parameter(torch.zeros(self.num_ropes, 256, self.hidden_size))
        
        self.load_specialists()
        # Default to bfloat16 for initial weights
        self.to(torch.bfloat16)
        self._is_gradient_checkpointing = False

    def load_specialists(self):
        sp_dir = os.path.join(self.checkpoint_dir, "scratchpads")
        os.makedirs(sp_dir, exist_ok=True)
        
        for i in range(self.num_layers):
            for j in range(self.num_ropes):
                path = os.path.join(sp_dir, f"L{i}_R{j}.pth")
                if os.path.exists(path):
                    self.specialist_grid[i][f"rope_{j}"].load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def save_specialists(self):
        sp_dir = os.path.join(self.checkpoint_dir, "scratchpads")
        for i in range(self.num_layers):
            for j in range(self.num_ropes):
                path = os.path.join(sp_dir, f"L{i}_R{j}.pth")
                torch.save(self.specialist_grid[i][f"rope_{j}"].state_dict(), path)

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)
        self._is_gradient_checkpointing = True

    def forward(self, input_ids, memory=None, cache_params=None, return_hidden=False, use_cache=True, cache_position=None):
        dtype = self.base_model.dtype
        device = input_ids.device
        
        # memory: [batch, num_ropes, num_pads, hidden]
        if memory is None:
            batch_size = input_ids.shape[0]
            memory = self.rope_init.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device).to(dtype)
        
        if cache_params is not None and cache_position is None:
            cache_position = torch.arange(input_ids.shape[1], device=device)

        hidden_states = self.base_model.backbone.embeddings(input_ids).to(dtype)
        
        for i, layer in enumerate(self.base_model.backbone.layers):
            router = self.routers[i]
            # 1. Causal Router selects Ropes
            routing_weights = torch.softmax(router(hidden_states), dim=-1).to(dtype) # [B, S, 8]
            
            # 2. Top-k Ropes (k=2)
            top_k_val, top_k_idx = torch.topk(routing_weights, k=2, dim=-1)
            mask = torch.zeros_like(routing_weights).scatter_(-1, top_k_idx, 1.0)
            routing_weights = routing_weights * mask
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            # 3. Mamba logic
            layer_outputs = layer(hidden_states, cache_params=cache_params, cache_position=cache_position)
            hidden_states = layer_outputs[0].to(dtype)
            
            # 4. Rope Interaction
            combined_retrieval = torch.zeros_like(hidden_states)
            new_memories = []
            
            # Identify which ropes are active in the entire batch for optimization
            active_ropes = torch.any(routing_weights > 0, dim=(0, 1))

            for j in range(self.num_ropes):
                rope_memory = memory[:, j]
                
                if active_ropes[j]:
                    sp_weight = routing_weights[:, :, j].unsqueeze(-1) # [B, S, 1]
                    specialist = self.specialist_grid[i][f"rope_{j}"]
                    
                    # Ensure alignment of B, S dimensions in case Mamba layer swapped them
                    if hidden_states.shape[0] != memory.shape[0]:
                        target_hs = hidden_states.transpose(0, 1)
                    else:
                        target_hs = hidden_states

                    if self.training and self._is_gradient_checkpointing:
                        h_out, m_out = torch.utils.checkpoint.checkpoint(
                            specialist, target_hs, rope_memory, sp_weight, use_reentrant=False
                        )
                    else:
                        h_out, m_out = specialist(target_hs, rope_memory, weight=sp_weight)
                    
                    # Combine retrieval. Note that h_out and hidden_states must have same shape.
                    # If we transposed, h_out will have different shape than hidden_states.
                    if hidden_states.shape[0] != memory.shape[0]:
                        combined_retrieval = combined_retrieval + (sp_weight * (h_out - target_hs)).transpose(0, 1)
                    else:
                        combined_retrieval = combined_retrieval + sp_weight * (h_out - hidden_states)
                    new_memories.append(m_out)
                else:
                    new_memories.append(rope_memory)
            
            # All ropes move to next layer
            memory = torch.stack(new_memories, dim=1)
            hidden_states = (hidden_states + combined_retrieval).to(dtype)
        
        hidden_states = self.base_model.backbone.norm_f(hidden_states).to(dtype)
        logits = self.base_model.lm_head(hidden_states)
        if return_hidden: return logits, hidden_states, cache_params, memory
        return logits, cache_params, memory

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, memory=None):
        generated = input_ids
        for i in range(max_new_tokens):
            # Fallback to O(N^2) for stability with custom MoS logic
            # This ensures correctness while we investigate MambaCache indexing
            logits, _, memory = self.forward(generated, memory=memory, use_cache=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == self.config.eos_token_id: break
        return generated
