import torch
import torch.nn as nn
import torch.utils.checkpoint
import os
from transformers import MambaForCausalLM, MambaConfig, AutoTokenizer
from transformers.models.mamba.modeling_mamba import MambaCache
from typing import Optional, Tuple, Union, List, Any

# --- Q4_0 Quantization Support ---

class Q4_0Linear(nn.Module):
    # ... (existing Q4_0Linear implementation) ...
    def __init__(self, in_features, out_features, bias=False, block_size=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def get_quantized_weight(self):
        w = self.weight
        out_f, in_f = w.shape
        if in_f % self.block_size != 0: return w
        w_reshaped = w.view(out_f, -1, self.block_size)
        scales = w_reshaped.abs().max(dim=-1, keepdim=True).values / 7.0
        q = torch.round(w_reshaped / (scales + 1e-6)).clamp(-8, 7)
        w_dequant = (q * scales).view(out_f, in_f)
        return w + (w_dequant - w).detach()

    def forward(self, x):
        w = self.get_quantized_weight()
        device, dtype = x.device, x.dtype
        bias = self.bias.to(device=device, dtype=dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w.to(device=device, dtype=dtype), bias)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class DynamicScratchpad(nn.Module):
    def __init__(self, hidden_size: int, num_pads: int = 128, rank: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pads = num_pads
        self.rank = rank
        
        # Rank-Space Projections
        self.in_proj = Q4_0Linear(hidden_size, rank * 3, bias=False) # Q, U, F
        self.out_proj = Q4_0Linear(rank, hidden_size, bias=False)
        
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Evolution happens in Rank-Space (High Efficiency)
        self.evolve = nn.Sequential(
            Q4_0Linear(rank, rank, bias=False),
            nn.GELU(),
            Q4_0Linear(rank, rank, bias=False)
        )
        self.evolve_gate = nn.Parameter(torch.zeros(1))
        self.diffusion_kernel = nn.Parameter(torch.eye(num_pads))
        self.mem_norm = RMSNorm(rank)

    def forward(self, x: torch.Tensor, memory_rank: torch.Tensor, weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # memory_rank: [batch, num_pads, rank]
        device = x.device
        if self.in_proj.weight.device != device:
            self.to(device)
            
        was_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            was_2d = True
            
        # 1. Retrieval (All in Rank-Space)
        projs = self.in_proj(x)
        q, u, f = torch.split(projs, self.rank, dim=-1)
        
        # Cross-attention in low-rank
        sim = torch.matmul(q, memory_rank.transpose(-1, -2)) # [B, S, Pads]
        attn = torch.softmax(sim / (self.rank**0.5), dim=-1)
        
        retrieved_rank = torch.matmul(attn, memory_rank) # [B, S, Rank]
        retrieved = self.out_proj(retrieved_rank)
        x_out = x + self.gate * retrieved
        
        # 2. Update (All in Rank-Space)
        f_gate = torch.sigmoid(f)
        if weight is not None:
             u = u * weight
             f_gate = f_gate * weight
             
        u_pads = torch.matmul(attn.transpose(-1, -2), u)
        f_pads = torch.matmul(attn.transpose(-1, -2), f_gate)
        
        # Direct rank-space update eliminates Hidden matmuls
        memory_rank = memory_rank * (1.0 - f_pads.clamp(0, 1)) + u_pads
        
        # Evolution & Diffusion in rank-space
        memory_rank = torch.matmul(self.diffusion_kernel, memory_rank)
        memory_rank = memory_rank + self.evolve_gate * self.evolve(memory_rank)
        memory_rank = self.mem_norm(memory_rank)
        
        if was_2d: x_out = x_out.squeeze(1)
        return x_out, memory_rank

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
    def __init__(self, hidden_size: int, num_pads: int = 128, rank: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pads = num_pads
        self.rank = rank
        
        # Low-rank bottleneck via Q4_0Linear
        self.in_proj = Q4_0Linear(hidden_size, rank * 3, bias=False) # Q, U, F
        self.out_proj = Q4_0Linear(rank, hidden_size, bias=False)
        
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Evolution happens in Rank-Space (High Efficiency)
        self.evolve = nn.Sequential(
            Q4_0Linear(rank, rank, bias=False),
            nn.GELU(),
            Q4_0Linear(rank, rank, bias=False)
        )
        self.evolve_gate = nn.Parameter(torch.zeros(1))
        self.diffusion_kernel = nn.Parameter(torch.eye(num_pads))
        self.mem_norm = RMSNorm(rank)

    def forward(self, x: torch.Tensor, memory_rank: torch.Tensor, weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # memory_rank: [batch, num_pads, rank]
        device = x.device
        if self.in_proj.weight.device != device:
            self.to(device)
            
        was_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            was_2d = True
            
        # 1. Retrieval (All in Rank-Space)
        projs = self.in_proj(x)
        q, u, f = torch.split(projs, self.rank, dim=-1)
        
        # Cross-attention in low-rank
        sim = torch.matmul(q, memory_rank.transpose(-1, -2)) # [B, S, Pads]
        attn = torch.softmax(sim / (self.rank**0.5), dim=-1)
        
        retrieved_rank = torch.matmul(attn, memory_rank) # [B, S, Rank]
        retrieved = self.out_proj(retrieved_rank)
        x_out = x + self.gate * retrieved
        
        # 2. Update (All in Rank-Space)
        f_gate = torch.sigmoid(f)
        if weight is not None:
             u = u * weight
             f_gate = f_gate * weight
             
        u_pads = torch.matmul(attn.transpose(-1, -2), u)
        f_pads = torch.matmul(attn.transpose(-1, -2), f_gate)
        
        # Direct rank-space update eliminates Hidden matmuls
        memory_rank = memory_rank * (1.0 - f_pads.clamp(0, 1)) + u_pads
        
        # Evolution & Diffusion in rank-space
        memory_rank = torch.matmul(self.diffusion_kernel, memory_rank)
        memory_rank = memory_rank + self.evolve_gate * self.evolve(memory_rank)
        memory_rank = self.mem_norm(memory_rank)
        
        if was_2d: x_out = x_out.squeeze(1)
        return x_out, memory_rank

class NeuxbaneThinking(nn.Module):
    def __init__(self, model_id_or_path: Optional[str] = None, checkpoint_dir: str = "checkpoint"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        if model_id_or_path:
            self.base_model = MambaForCausalLM.from_pretrained(model_id_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            # 1B-Base Mamba Config (Standard is ~48 layers, here 32 layers x 2048 hidden leads to ~1B)
            config = MambaConfig(
                vocab_size=50260, 
                hidden_size=2048, 
                state_size=16, 
                num_hidden_layers=32, 
                expand=2, 
                conv_kernel=4, 
                use_cache=True, 
                rms_norm_eps=1e-5, 
                torch_dtype=torch.bfloat16
            )
            self.base_model = MambaForCausalLM(config)
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_ropes = 4 # Total specialists = 32 layers * 4 ropes = 128
        
        # Specialist Grid: num_layers x num_ropes
        self.specialist_grid = nn.ModuleList([
            nn.ModuleDict({
                f"rope_{j}": DynamicScratchpad(self.hidden_size, num_pads=128) for j in range(self.num_ropes)
            }) for i in range(self.num_layers)
        ])
        
        # Routers: One per layer deciding which "rope" to pull
        self.routers = nn.ModuleList([nn.Linear(self.hidden_size, self.num_ropes) for _ in range(self.num_layers)])
        
        # Per-rope initial memory in Rank-Space
        self.rope_init = nn.Parameter(torch.zeros(self.num_ropes, 128, 32)) # Rank=32
        
        self.load_specialists()
        # Default to bfloat16 for initial weights
        self.to(torch.bfloat16)
        
        # Expert Offload JIT: Move all experts to CPU initially
        self.offload_specialists()
        self._is_gradient_checkpointing = False

    def offload_specialists(self):
        """Force all experts to CPU to save GPU RAM."""
        for layer_ropes in self.specialist_grid:
            for rope in layer_ropes.values():
                rope.to("cpu")

    def to(self, *args, **kwargs):
        """Override to ensure specialists stay on CPU while base model moves to GPU."""
        target_dtype = None
        for arg in args:
            if isinstance(arg, torch.dtype): target_dtype = arg
        if 'dtype' in kwargs: target_dtype = kwargs['dtype']

        if target_dtype:
            for layer_ropes in self.specialist_grid:
                for rope in layer_ropes.values():
                    rope.to(dtype=target_dtype)

        grid = self._modules.pop('specialist_grid')
        try:
            super().to(*args, **kwargs)
        finally:
            self._modules['specialist_grid'] = grid
        return self

    def load_specialists(self):
        sp_dir = os.path.join(self.checkpoint_dir, "scratchpads")
        os.makedirs(sp_dir, exist_ok=True)
        
        for i in range(self.num_layers):
            for j in range(self.num_ropes):
                path = os.path.join(sp_dir, f"L{i}_R{j}.pth")
                if os.path.exists(path):
                    try:
                        state_dict = torch.load(path, map_location="cpu", weights_only=True)
                        self.specialist_grid[i][f"rope_{j}"].load_state_dict(state_dict, strict=True)
                    except Exception as e:
                        print(f"Skipping specialist L{i}_R{j} due to loading error: {e}")

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
        
        # memory: [batch, num_ropes, num_pads, rank]
        if memory is None:
            batch_size = input_ids.shape[0]
            memory = self.rope_init.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device).to(dtype)
        
        total_aux_loss = 0.0

        if cache_params is not None and cache_position is None:
            cache_position = torch.arange(input_ids.shape[1], device=device)

        hidden_states = self.base_model.backbone.embeddings(input_ids).to(dtype)
        
        for i, layer in enumerate(self.base_model.backbone.layers):
            router = self.routers[i]
            
            # Causal Routing + Stability Losses
            router_logits = router(hidden_states).to(torch.float32)
            routing_weights = torch.softmax(router_logits, dim=-1).to(dtype) 
            
            # 1. Router Z-Loss (Stability)
            z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1)**2)
            
            # 2. Auxiliary Load Balancing Loss
            freqs = routing_weights.mean(dim=(0, 1))
            probs = torch.softmax(router_logits.mean(dim=(0, 1)), dim=-1)
            aux_loss = self.num_ropes * torch.sum(freqs * probs)
            total_aux_loss = total_aux_loss + 0.01 * (z_loss * 0.1 + aux_loss)

            # Top-k Ropes (k=2)
            top_k_val, top_k_idx = torch.topk(routing_weights, k=2, dim=-1)
            mask = torch.zeros_like(routing_weights).scatter_(-1, top_k_idx, 1.0)
            routing_weights = routing_weights * mask
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            # Mamba logic
            layer_outputs = layer(hidden_states, cache_params=cache_params, cache_position=cache_position)
            hidden_states = layer_outputs[0].to(dtype)
            
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            combined_retrieval = torch.zeros_like(hidden_states)
            new_memories = []
            active_ropes = torch.any(routing_weights > 0, dim=(0, 1))

            # Fetch active specialists for this layer
            for j in range(self.num_ropes):
                if active_ropes[j]:
                    self.specialist_grid[i][f"rope_{j}"].to(device, non_blocking=True)

            for j in range(self.num_ropes):
                rope_memory = memory[:, j]
                
                if active_ropes[j]:
                    sp_weight = routing_weights[:, :, j].unsqueeze(-1)
                    specialist = self.specialist_grid[i][f"rope_{j}"]
                    
                    target_hs = hidden_states
                    if hidden_states.shape[0] != memory.shape[0] and hidden_states.dim() == 3:
                        target_hs = hidden_states.transpose(0, 1)

                    if self.training and self._is_gradient_checkpointing:
                        h_out, m_out = torch.utils.checkpoint.checkpoint(
                            specialist, target_hs, rope_memory, sp_weight, use_reentrant=False
                        )
                    else:
                        h_out, m_out = specialist(target_hs, rope_memory, weight=sp_weight)
                    
                    delta = sp_weight * (h_out - target_hs)
                    if hidden_states.shape[0] != memory.shape[0] and hidden_states.dim() == 3:
                        combined_retrieval = combined_retrieval + delta.transpose(0, 1)
                    else:
                        combined_retrieval = combined_retrieval + delta
                    new_memories.append(m_out)
                else:
                    new_memories.append(rope_memory)
            
            # Post-Layer Offload
            for j in range(self.num_ropes):
                if active_ropes[j]:
                    self.specialist_grid[i][f"rope_{j}"].to("cpu", non_blocking=True)
            
            memory = torch.stack(new_memories, dim=1)
            hidden_states = (hidden_states + combined_retrieval).to(dtype)
        
        hidden_states = self.base_model.backbone.norm_f(hidden_states).to(dtype)
        logits = self.base_model.lm_head(hidden_states)
        
        if return_hidden: return logits, hidden_states, cache_params, memory, total_aux_loss
        return logits, cache_params, memory, total_aux_loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, memory=None):
        generated = input_ids
        for i in range(max_new_tokens):
            logits, _, memory, _ = self.forward(generated, memory=memory, use_cache=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == self.config.eos_token_id: break
        return generated
