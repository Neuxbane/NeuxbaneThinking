import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import os
import math
from transformers import AutoTokenizer
from typing import Optional, Tuple, Union, List, Any

# --- Q4_0 Quantization Support ---

class Q4_0Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, block_size=32):
        super().__init__()
        self.in_features, self.out_features, self.block_size = in_features, out_features, block_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias: self.bias = nn.Parameter(torch.zeros(out_features))
        else: self.register_parameter("bias", None)
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
        return F.linear(x, w.to(device=device, dtype=dtype), bias)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# --- Custom State Space Model (SSM) ---

class CustomSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_expand=2, d_conv=4, layer_idx=0):
        super().__init__()
        self.d_model, self.d_state, self.d_expand, self.layer_idx = d_model, d_state, d_expand, layer_idx
        self.d_inner, self.d_conv = d_model * d_expand, d_conv
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Corrected A initialization: log of positive values
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, cache_params=None, cache_position=None):
        B, S, _ = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        if cache_params is not None and S == 1:
            conv_state = cache_params.conv_states[self.layer_idx]
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x_proj.squeeze(1)
            x_conv = torch.sum(conv_state * self.conv.weight.squeeze(1), dim=-1).unsqueeze(1)
        else:
            x_conv = self.conv(x_proj.transpose(1, 2))[:, :, :S].transpose(1, 2)
            if cache_params is not None:
                last_tokens = x_proj.transpose(1, 2)[:, :, -self.d_conv:]
                if last_tokens.shape[-1] < self.d_conv: last_tokens = F.pad(last_tokens, (self.d_conv - last_tokens.shape[-1], 0))
                cache_params.conv_states[self.layer_idx].copy_(last_tokens)
        
        x_conv = F.silu(x_conv)
        proj = self.x_proj(x_conv)
        dt, B_vals, C_vals = torch.split(proj, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).to(torch.float32)
        A = -torch.exp(self.A_log).to(torch.float32) 
        
        if cache_params is not None and S == 1:
            h = cache_params.ssm_states[self.layer_idx].to(torch.float32)
            dA = torch.exp(dt[:, 0].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, 0].unsqueeze(-1) * B_vals[:, 0].unsqueeze(1)
            h = dA * h + dB * x_conv[:, 0].unsqueeze(-1)
            cache_params.ssm_states[self.layer_idx].copy_(h)
            y = torch.sum(h * C_vals[:, 0].unsqueeze(1), dim=-1).unsqueeze(1)
        else:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
            outputs = []
            for i in range(S):
                dA = torch.exp(dt[:, i].unsqueeze(-1) * A.unsqueeze(0))
                dB = dt[:, i].unsqueeze(-1) * B_vals[:, i].unsqueeze(1)
                h = dA * h + dB * x_conv[:, i].unsqueeze(-1)
                outputs.append(torch.sum(h * C_vals[:, i].unsqueeze(1), dim=-1))
            y = torch.stack(outputs, dim=1)
            if cache_params is not None: cache_params.ssm_states[self.layer_idx].copy_(h)
        return self.out_proj(((y + x_conv * self.D) * F.silu(z)).to(xz.dtype))

class MambaCache:
    def __init__(self, config, batch_size, device, dtype):
        self.conv_states = [torch.zeros(batch_size, config.hidden_size * config.expand, config.conv_kernel, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.ssm_states = [torch.zeros(batch_size, config.hidden_size * config.expand, config.state_size, device=device, dtype=torch.float32) for _ in range(config.num_hidden_layers)]

class MambaLayer(nn.Module):
    def __init__(self, d_model, config, layer_idx):
        super().__init__()
        self.norm, self.mixer = RMSNorm(d_model), CustomSSM(d_model, config.state_size, config.expand, config.conv_kernel, layer_idx)
    def forward(self, x, cache_params=None, cache_position=None): return x + self.mixer(self.norm(x), cache_params, cache_position)

class MambaConfig:
    def __init__(self, vocab_size=50257, hidden_size=768, num_hidden_layers=24, state_size=16, expand=2, conv_kernel=4, rms_norm_eps=1e-5, **kwargs):
        self.vocab_size, self.hidden_size, self.num_hidden_layers = vocab_size, hidden_size, num_hidden_layers
        self.state_size, self.expand, self.conv_kernel, self.rms_norm_eps = state_size, expand, conv_kernel, rms_norm_eps
        for k, v in kwargs.items(): setattr(self, k, v)

class DynamicScratchpad(nn.Module):
    def __init__(self, hidden_size: int, num_pads: int = 128, rank: int = 32):
        super().__init__()
        self.hidden_size, self.num_pads, self.rank = hidden_size, num_pads, rank
        self.in_proj, self.out_proj = Q4_0Linear(hidden_size, rank * 3, bias=False), Q4_0Linear(rank, hidden_size, bias=False)
        self.gate, self.evolve_gate = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.evolve = nn.Sequential(Q4_0Linear(rank, rank, bias=False), nn.GELU(), Q4_0Linear(rank, rank, bias=False))
        self.diffusion_kernel, self.mem_norm = nn.Parameter(torch.eye(num_pads)), RMSNorm(rank)

    def forward(self, x, memory_rank, weight=None):
        if self.in_proj.weight.device != x.device: self.to(x.device)
        was_2d = False
        if x.dim() == 2: x, was_2d = x.unsqueeze(1), True
        q, u, f = torch.split(self.in_proj(x), self.rank, dim=-1)
        attn = torch.softmax(torch.matmul(q, memory_rank.transpose(-1, -2)) / (self.rank**0.5), dim=-1)
        x_out = x + self.gate * self.out_proj(torch.matmul(attn, memory_rank))
        f_gate = torch.sigmoid(f)
        if weight is not None: u, f_gate = u * weight, f_gate * weight
        memory_rank = memory_rank * (1.0 - torch.matmul(attn.transpose(-1, -2), f_gate).clamp(0, 1)) + torch.matmul(attn.transpose(-1, -2), u)
        memory_rank = self.mem_norm(torch.matmul(self.diffusion_kernel, memory_rank) + self.evolve_gate * self.evolve(memory_rank))
        return (x_out.squeeze(1) if was_2d else x_out), memory_rank

class BPETokenizer:
    def __init__(self, model_id="openai-community/gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>", "<user>", "<assistant>"]})
        self.vocab_size = len(self.tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.eos_token_id, self.user_token_id, self.assistant_token_id = self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<user>"), self.tokenizer.convert_tokens_to_ids("<assistant>")
    def encode(self, text, add_special_tokens=True, max_length=None): return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_length)
    def decode(self, token_ids): return self.tokenizer.decode(token_ids)

class NeuxbaneSSM(nn.Module):
    def __init__(self, model_id_or_path=None, checkpoint_dir="checkpoint"):
        super().__init__()
        self.checkpoint_dir, self.tokenizer_bpe = checkpoint_dir, BPETokenizer()
        vocab_size = self.tokenizer_bpe.vocab_size
        self.config = MambaConfig(vocab_size=vocab_size, hidden_size=768, num_hidden_layers=24, state_size=16, expand=2, conv_kernel=4)
        self.embeddings = nn.Embedding(vocab_size, 768)
        self.layers = nn.ModuleList([MambaLayer(768, self.config, i) for i in range(24)])
        self.norm_f = RMSNorm(768)
        self.lm_head = nn.Linear(768, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.weight
        self.specialist_grid = nn.ModuleList([nn.ModuleDict({f"rope_{j}": DynamicScratchpad(768) for j in range(4)}) for i in range(24)])
        self.routers = nn.ModuleList([nn.Linear(768, 4) for _ in range(24)])
        self.rope_init = nn.Parameter(torch.zeros(4, 128, 32))
        self.load_specialists(); self.to(torch.bfloat16); self._is_gradient_checkpointing = False

    def load_specialists(self):
        path = os.path.join(self.checkpoint_dir, "specialists.pt")
        if os.path.exists(path):
            try: self.specialist_grid.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            except: pass

    def save_specialists(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.specialist_grid.state_dict(), os.path.join(self.checkpoint_dir, "specialists.pt"))

    def gradient_checkpointing_enable(self, **kwargs): self._is_gradient_checkpointing = True

    def _forward_layer(self, layer_idx, hidden_states, memory, cache_params, cache_position):
        h_mamba = self.layers[layer_idx](hidden_states, cache_params, cache_position)
        router_logits = self.routers[layer_idx](h_mamba).to(torch.float32)
        routing_weights = torch.softmax(router_logits, dim=-1).to(hidden_states.dtype)
        top_k_idx = torch.topk(routing_weights, k=2, dim=-1).indices
        mask = torch.zeros_like(routing_weights).scatter_(-1, top_k_idx, 1.0)
        routing_weights = (routing_weights * mask) / (routing_weights * mask).sum(dim=-1, keepdim=True).clamp(1e-6)
        new_mems, combined_delta = [], torch.zeros_like(h_mamba)
        for j in range(4):
            h_out, m_out = self.specialist_grid[layer_idx][f"rope_{j}"](h_mamba, memory[:, j], routing_weights[:, :, j:j+1])
            combined_delta += routing_weights[:, :, j:j+1] * (h_out - h_mamba)
            new_mems.append(m_out)

        # Re-introduce aux loss for router stability
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1)**2)
        freqs = routing_weights.mean(dim=(0, 1))
        # Probability distribution of the router
        probs = torch.softmax(router_logits.mean(dim=(0, 1)), dim=-1)
        # Load balancing loss (MoE style)
        aux_loss = 4.0 * torch.sum(freqs * probs) # Normalize by num_ropes
        total_aux = 0.01 * (z_loss * 0.1 + aux_loss)

        return h_mamba + combined_delta, torch.stack(new_mems, dim=1), total_aux

    def forward(self, input_ids, memory=None, cache_params=None, return_hidden=False, use_cache=False, cache_position=None):
        dtype, device = self.lm_head.weight.dtype, input_ids.device
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        B, S = input_ids.shape
        if memory is None: memory = self.rope_init.unsqueeze(0).repeat(B, 1, 1, 1).to(device).to(dtype)
        hidden_states, total_aux = self.embeddings(input_ids).to(dtype), 0.0
        if use_cache and cache_params is None: cache_params = MambaCache(self.config, B, device, dtype)
        for i in range(24):
            if self.training and self._is_gradient_checkpointing:
                hidden_states, memory, layer_aux = torch.utils.checkpoint.checkpoint(self._forward_layer, i, hidden_states, memory, cache_params, cache_position, use_reentrant=False)
            else:
                hidden_states, memory, layer_aux = self._forward_layer(i, hidden_states, memory, cache_params, cache_position)
            total_aux += layer_aux
        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states)
        if return_hidden: return logits, cache_params, memory, total_aux, hidden_states
        return logits, cache_params, memory, total_aux

    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9):
        self.eval()
        current_ids, memory, cache_params = input_ids, None, None
        for i in range(max_new_tokens):
            with torch.no_grad():
                step_ids = current_ids if i == 0 else current_ids[:, -1:]
                logits, cache_params, memory, _ = self.forward(step_ids, memory, cache_params, use_cache=True)
                next_token_logits = logits[:, -1, :] / (temperature + 1e-6)
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = (cumulative_probs > top_p); sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone(); sorted_indices_to_remove[..., 0] = 0
                next_token_logits[sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)] = -float('Inf')
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer_bpe.eos_token_id: break
        return current_ids
