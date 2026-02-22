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

class NeuxbaneThinking(nn.Module):
    def __init__(self, model_id_or_path: Optional[str] = None, checkpoint_dir: str = "checkpoint"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        if model_id_or_path:
            self.base_model = MambaForCausalLM.from_pretrained(model_id_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            config = MambaConfig(
                vocab_size=50260, 
                hidden_size=768, 
                state_size=16,
                num_hidden_layers=24, 
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
        self.num_ropes = 4 
        
        self.specialist_grid = nn.ModuleList([
            nn.ModuleDict({
                f"rope_{j}": DynamicScratchpad(self.hidden_size, num_pads=128) for j in range(self.num_ropes)
            }) for i in range(self.num_layers)
        ])
        
        self.routers = nn.ModuleList([nn.Linear(self.hidden_size, self.num_ropes) for _ in range(self.num_layers)])
        self.rope_init = nn.Parameter(torch.zeros(self.num_ropes, 128, 32)) 
        
        self.load_specialists()
        self.to(torch.bfloat16)
        self._is_gradient_checkpointing = False

    def load_specialists(self):
        path = os.path.join(self.checkpoint_dir, "specialists.pt")
        if not os.path.exists(path): return
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            self.specialist_grid.load_state_dict(state_dict)
            print(f"Loaded {self.num_layers*self.num_ropes} specialists from {path}")
        except Exception as e:
            print(f"Failed to load specialists: {e}")

    def save_specialists(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "specialists.pt")
        torch.save(self.specialist_grid.state_dict(), path)
        print(f"Saved {self.num_layers*self.num_ropes} specialists to {path}")

    def gradient_checkpointing_enable(self, **kwargs):
        # self.base_model.gradient_checkpointing_enable(**kwargs) # Skip base model GC as we do it manually per _forward_layer
        self._is_gradient_checkpointing = True

    def _forward_layer(self, layer_idx, hidden_states, memory, cache_params, cache_position):
        layer = self.base_model.backbone.layers[layer_idx]
        router = self.routers[layer_idx]
        dtype = hidden_states.dtype
        B, S, D = hidden_states.shape
        
        # 1. Mamba Layer
        mamba_outputs = layer(hidden_states, cache_params=cache_params, cache_position=cache_position)
        h_mamba = mamba_outputs[0].to(dtype).view(B, S, D)
        
        # 2. Routing Logic
        router_logits = router(h_mamba).to(torch.float32)
        routing_weights = torch.softmax(router_logits, dim=-1).to(dtype)
        
        # Top-2 routing
        top_k_val, top_k_idx = torch.topk(routing_weights, k=2, dim=-1)
        mask = torch.zeros_like(routing_weights).scatter_(-1, top_k_idx, 1.0)
        routing_weights = routing_weights * mask
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 3. Specialist Interaction
        new_mems = []
        combined_delta = torch.zeros_like(h_mamba)
        
        rw = routing_weights
        
        for j in range(self.num_ropes):
            specialist = self.specialist_grid[layer_idx][f"rope_{j}"]
            rope_memory = memory[:, j] # [B, Pads, Rank]
            sp_weight = rw[:, :, j].unsqueeze(-1) # [B, S, 1]
            
            h_out, m_out = specialist(h_mamba, rope_memory, weight=sp_weight)
            combined_delta = combined_delta + sp_weight * (h_out - h_mamba)
            new_mems.append(m_out)
            
        h_final = (h_mamba + combined_delta).to(dtype)
        memory_final = torch.stack(new_mems, dim=1)
        
        # Stability losses (Aux loss)
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1)**2)
        freqs = routing_weights.mean(dim=(0, 1))
        probs = torch.softmax(router_logits.mean(dim=(0, 1)), dim=-1)
        aux_loss = self.num_ropes * torch.sum(freqs * probs)
        total_loss = 0.01 * (z_loss * 0.1 + aux_loss)
        
        return h_final, memory_final, total_loss

    def forward(self, input_ids, memory=None, cache_params=None, return_hidden=False, use_cache=False, cache_position=None):
        dtype = self.base_model.dtype
        device = input_ids.device
        
        # Ensure 2D input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        B = input_ids.shape[0]
        S = input_ids.shape[1]
        
        if memory is None:
            # Memory should be [B, Ropes, Pads, Rank]
            memory = self.rope_init.unsqueeze(0).repeat(B, 1, 1, 1).to(device).to(dtype)
        
        total_aux_loss = 0.0
        hidden_states = self.base_model.backbone.embeddings(input_ids).to(dtype)
        
        # Position mask for Mamba (Automatic handled by layers if None)
        if cache_params is not None and cache_position is None:
             # Assume S is the new token count, and calculate position for it
             # Wait, MambaCache doesn't track seen_tokens.
             # So we let the caller pass cache_position if they have it.
             pass
            
        for i in range(self.num_layers):
            if self.training and self._is_gradient_checkpointing:
                hidden_states, memory, layer_aux = torch.utils.checkpoint.checkpoint(
                    self._forward_layer, i, hidden_states, memory, cache_params, cache_position,
                    use_reentrant=False
                )
            else:
                hidden_states, memory, layer_aux = self._forward_layer(
                    i, hidden_states, memory, cache_params, cache_position
                )
            # FORCE [B, S, D]
            hidden_states = hidden_states.view(B, S, -1)
            total_aux_loss = total_aux_loss + layer_aux

        hidden_states = self.base_model.backbone.norm_f(hidden_states).to(dtype)
        logits = self.base_model.lm_head(hidden_states)
        logits = logits.view(B, S, -1)
        
        if return_hidden: return logits, hidden_states, cache_params, memory, total_aux_loss
        return logits, cache_params, memory, total_aux_loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, memory=None):
        self.eval()
        generated = input_ids
        for i in range(max_new_tokens):
            logits, _, memory, _ = self.forward(generated, memory=memory, use_cache=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == self.config.eos_token_id: break
        return generated
