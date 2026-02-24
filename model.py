import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import re

class ByteTokenizer:
    def __init__(self):
        self.special_tokens = [
            "<think>", "</think>", 
            "<role>", "</role>", 
            "<eos>", "<bos>", 
            "<tools>", "</tools>",
            "<tool_calls>", "</tool_calls>"
        ]
        # Mapping for special tokens starting from 256
        self.special_to_id = {t: 256 + i for i, t in enumerate(self.special_tokens)}
        self.id_to_special = {i: t for t, i in self.special_to_id.items()}
        self.vocab_size = 256 + len(self.special_tokens)
        
        # Regex for matching special tokens or any single character
        pattern = "|".join(re.escape(t) for t in self.special_tokens)
        self.regex = re.compile(f"({pattern})")

    def encode(self, text):
        parts = self.regex.split(text)
        ids = []
        for part in parts:
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(list(part.encode('utf-8')))
        return ids

    def decode(self, ids):
        out_bytes = bytearray()
        result = ""
        for i in ids:
            if i in self.id_to_special:
                if out_bytes:
                    result += out_bytes.decode('utf-8', errors='replace')
                    out_bytes = bytearray()
                result += self.id_to_special[i]
            elif 0 <= i < 256:
                out_bytes.append(i)
        
        if out_bytes:
            result += out_bytes.decode('utf-8', errors='replace')
        return result

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # We support Grouped-Query Attention (GQA) where multiple queries share a single KV head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, (self.n_head + 2 * self.n_kv_head) * self.head_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # RoPE precomputation
        # Total effective sequence length must include the prefix scratchpad
        # Using active memory slots as the effective scratchpad
        n_sp = config.n_active_slots * config.n_slot_len
        effective_block_size = config.block_size + n_sp
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(effective_block_size).float()
        freqs = torch.outer(t, inv_freq) # (effective_block_size, head_dim // 2)
        self.register_buffer("cos", freqs.cos().view(1, 1, effective_block_size, self.head_dim // 2))
        self.register_buffer("sin", freqs.sin().view(1, 1, effective_block_size, self.head_dim // 2))

        # flash attention uses internal causal mask when is_causal=True
        # but we keep bias for cases where manual masking might be needed
        self.register_buffer("bias", torch.tril(torch.ones(effective_block_size, effective_block_size))
                                     .view(1, 1, effective_block_size, effective_block_size))

    def _apply_rope(self, x, start_pos=0):
        B, nh, T, hs = x.size()
        cos = self.cos[:, :, start_pos:start_pos+T, :]
        sin = self.sin[:, :, start_pos:start_pos+T, :]
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, x, kv_cache=None, start_pos_offset=0):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values
        # q: (B, T, n_head * head_dim), k: (B, T, n_kv_head * head_dim), v: (B, T, n_kv_head * head_dim)
        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_head * self.head_dim
        q, k, v  = self.c_attn(x).split([q_size, kv_size, kv_size], dim=2)
        
        # Reshape to multi-head format
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # (B, n_kv_head, T, head_dim)

        start_pos = start_pos_offset
        if kv_cache is not None:
            # kv_cache[0].shape[2] represents the number of already cached tokens
            start_pos += kv_cache[0].shape[2]
            
            # Bound context to avoid out-of-range on precomputed RoPE/masks
            max_seq_len = self.cos.size(2)
            if start_pos + T > max_seq_len:
                excess = start_pos + T - max_seq_len
                prev_k, prev_v = kv_cache
                kv_cache = (prev_k[:, :, excess:, :], prev_v[:, :, excess:, :])
                start_pos -= excess
            
        # rotate queries and keys using their positions
        q = self._apply_rope(q, start_pos=start_pos)
        k = self._apply_rope(k, start_pos=start_pos)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        new_kv_cache = (k, v)
        
        # repeat KV heads if n_kv_head < n_head
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, k.size(2), self.head_dim).reshape(B, self.n_head, k.size(2), self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, v.size(2), self.head_dim).reshape(B, self.n_head, v.size(2), self.head_dim)
        
        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1 and kv_cache is None))
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.c_proj(y)
        return y, new_kv_cache

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU activation; using 4 * n_embd as intermediate dim
        # We combine the gate and value projections into one c_fc for efficiency
        self.c_fc    = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x, gate = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(x) * gate
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, use_memory_read=False):
        super().__init__()
        self.config = config
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.memory_read = MemoryRead(config) if use_memory_read else None

    def forward(self, x, kv_cache=None, start_pos_offset=0, memory_pool=None):
        # Self-Attention handles prefix memory naturally via causal mask if prepended
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, start_pos_offset=start_pos_offset)
        x = x + attn_out
        
        if self.memory_read is not None:
            x = self.memory_read(x, memory_pool)
            
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

class MemoryRead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_in = RMSNorm(config.n_embd)
        self.ln_out = RMSNorm(config.n_embd)
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, memory_pool):
        if memory_pool is None:
            return x
        
        b, t, c = x.size()
        m_slots, m_len, _ = memory_pool.size()
        
        # Flatten memory for K, V
        mem = memory_pool.view(1, m_slots * m_len, c).expand(b, -1, -1)
        
        q = self.q_proj(self.ln_in(x)).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(mem).view(b, m_slots * m_len, 2, self.n_head, self.head_dim)
        k = kv[:, :, 0, :, :].transpose(1, 2) 
        v = kv[:, :, 1, :, :].transpose(1, 2) 
        
        # Cross-attention (no causal mask since it's memory)
        y = F.scaled_dot_product_attention(q, k, v)
        
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return x + self.ln_out(self.out_proj(y))

class TransformerConfig:
    def __init__(self, vocab_size=256, block_size=512, n_layer=6, n_head=6, n_kv_head=None, n_embd=384, 
                 n_memory_slots=64, n_slot_len=128, n_active_slots=8, rope_theta=10000.0, router_temp=1.0,
                 use_delta_rule=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_embd = n_embd
        self.n_memory_slots = n_memory_slots
        self.n_slot_len = n_slot_len
        self.n_active_slots = n_active_slots
        self.rope_theta = rope_theta
        self.router_temp = router_temp
        self.use_delta_rule = use_delta_rule

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, use_memory_read=(i % 3 == 2)) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.n_memory_slots > 0:
            # initialization for memory pool and lookup keys
            # Using learned constants for better memory startup (Zero-init with bias or small noise)
            self.memory_pool = nn.Parameter(torch.zeros(config.n_memory_slots, config.n_slot_len, config.n_embd))
            self.memory_pos_emb = nn.Parameter(torch.randn(1, config.n_slot_len, config.n_embd) * 0.02)
            self.memory_keys = nn.Parameter(torch.randn(config.n_memory_slots, config.n_embd) * 0.02)
            
            # Query-based Routing (Instead of just x.mean())
            self.router_query_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.router_ln = RMSNorm(config.n_embd)
            
            # Dynamic Update Head (The "Write" mechanism)
            head_dim = config.n_embd // config.n_head
            self.memory_updater = nn.Linear(config.n_embd, config.n_active_slots * 2 * config.n_kv_head * head_dim)
            self.memory_gate = nn.Linear(config.n_embd, config.n_active_slots)
        else:
            self.memory_pool = None

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        # subtract tied weight
        n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None, kv_caches=None, active_indices=None, is_thinking=None):
        device = idx.device
        b, t = idx.size()
        
        # Effective n_sp for RoPE/mask indices
        n_sp = self.config.n_active_slots * self.config.n_slot_len
        
        if kv_caches is None:
            # Prefill / Initial forward pass
            if t > self.config.block_size:
                t = self.config.block_size
                idx = idx[:, -t:]
                if targets is not None:
                    targets = targets[:, -t:]

            x = self.transformer.wte(idx) 
            if n_sp > 0:
                # Top-K Scratchpad Routing
                # Match current sequence mean embedding against the memory pool's keys
                query = self.router_ln(self.router_query_proj(x.mean(dim=1))) # (b, n_embd)
                
                # similarity scores: (b, n_memory_slots)
                sim = torch.matmul(query, self.memory_keys.t())
                
                base_mem = self.memory_pool + self.memory_pos_emb
                if self.training:
                    # Soft Routing during training for exploration/gradients
                    probs = torch.softmax(sim / self.config.router_temp, dim=-1)
                    # Weighted sum of all slots: (b, n_slot_len, n_embd)
                    soft_prefix = (probs.view(b, -1, 1, 1) * base_mem.unsqueeze(0)).sum(dim=1)
                    # Expand to match n_active_slots for sequence length consistency
                    selected_slots = soft_prefix.unsqueeze(1).repeat(1, self.config.n_active_slots, 1, 1)
                    _, active_indices = torch.topk(sim, self.config.n_active_slots, dim=-1)
                else:
                    # Pick the most relevant TOP memory slots
                    _, active_indices = torch.topk(sim, self.config.n_active_slots, dim=-1) # (b, n_active_slots)
                    selected_slots = base_mem[active_indices] 
                
                # prefix: (b, n_sp, n_embd)
                prefix = selected_slots.contiguous().view(b, n_sp, self.config.n_embd)
                
                x = torch.cat([prefix, x], dim=1)
                if targets is not None:
                    # Offset targets to align with sequence tokens
                    target_pad = torch.full((b, n_sp), -1, dtype=targets.dtype, device=targets.device)
                    targets = torch.cat([target_pad, targets], dim=1)
            kv_caches = [None] * len(self.transformer.h)
        else:
            # Incremental pass: kv_caches already contains the prefix
            x = self.transformer.wte(idx)

        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            x, cache = block(x, kv_cache=kv_caches[i], start_pos_offset=0, memory_pool=self.memory_pool)
            new_kv_caches.append(cache)
            
        x_post = self.transformer.ln_f(x)
        logits = self.lm_head(x_post)

        # --- Dynamic Memory Update (Write-Back) ---
        # At each step, update the "Active Scratchpad" (Prefix KV) using current thoughts
        # Interleaved Thinking: Only update when is_thinking is True
        should_update = (self.config.n_memory_slots > 0 and 
                        active_indices is not None and 
                        t == 1 and 
                        (is_thinking is None or is_thinking.any()))
        
        if should_update:
            current_h = x_post[:, -1, :] # (b, n_embd)
            head_dim = self.config.n_embd // self.config.n_head
            
            updates = self.memory_updater(current_h).view(b, self.config.n_active_slots, 2, self.config.n_kv_head, head_dim)
            k_up = updates[:, :, 0, :, :]
            v_up = updates[:, :, 1, :, :]
            
            # Predict a gate for each active slot
            raw_gates = self.memory_gate(current_h) # (b, n_active_slots)
            
            # Sparse Update: Only update the top-k most relevant active slots
            # This prevents gradients from washing out across all slots.
            k_sparse = max(1, self.config.n_active_slots // 2)
            top_gates, top_indices = torch.topk(raw_gates, k_sparse, dim=-1)
            
            # Create a sparse gate mask
            sparse_mask = torch.zeros_like(raw_gates).scatter_(-1, top_indices, 1.0)
            gates = torch.sigmoid(raw_gates) * sparse_mask
            gates = gates.view(b, self.config.n_active_slots, 1)
            
            # Mask based on is_thinking if provided
            if is_thinking is not None:
                gates = gates * is_thinking.view(b, 1, 1).float()

            for i in range(len(self.transformer.h)):
                k, v = new_kv_caches[i]
                # Update the K and V of the prefix section (first n_sp tokens)
                prefix_k = k[:, :, :n_sp, :].clone().view(b, self.config.n_kv_head, self.config.n_active_slots, self.config.n_slot_len, head_dim)
                prefix_v = v[:, :, :n_sp, :].clone().view(b, self.config.n_kv_head, self.config.n_active_slots, self.config.n_slot_len, head_dim)
                
                # Expand updates to match heads and slot length
                k_update = k_up.transpose(1, 2).unsqueeze(3)
                v_update = v_up.transpose(1, 2).unsqueeze(3)
                g = gates.view(b, 1, self.config.n_active_slots, 1, 1)
                
                if self.config.use_delta_rule:
                    # Delta Rule: error = update - current, current = current + g * error
                    prefix_k = prefix_k + g * (k_update - prefix_k)
                    prefix_v = prefix_v + g * (v_update - prefix_v)
                else:
                    # Recurrent update for both K and V
                    prefix_k = (1 - g) * prefix_k + g * k_update
                    prefix_v = (1 - g) * prefix_v + g * v_update
                
                # Re-integrate into cache
                prefix_k = prefix_k.view(b, self.config.n_kv_head, n_sp, head_dim)
                prefix_v = prefix_v.view(b, self.config.n_kv_head, n_sp, head_dim)
                new_kv_caches[i] = (torch.cat([prefix_k, k[:, :, n_sp:, :]], dim=2), 
                                   torch.cat([prefix_v, v[:, :, n_sp:, :]], dim=2))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return logits, loss, new_kv_caches, active_indices

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        full_idx = idx
        kv_caches = None
        active_indices = None
        
        # Track whether the model is in thinking mode (starting with the last token's state)
        # 256: <think>, 257: </think>
        is_thinking = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for b_idx in range(idx.size(0)):
            # Find the last <think> or </think> in the input
            think_pos = (idx[b_idx] == 256).nonzero(as_tuple=True)[0]
            unthink_pos = (idx[b_idx] == 257).nonzero(as_tuple=True)[0]
            last_think = think_pos[-1] if len(think_pos) > 0 else -1
            last_unthink = unthink_pos[-1] if len(unthink_pos) > 0 else -1
            if last_think > last_unthink:
                is_thinking[b_idx] = True

        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _, kv_caches, active_indices = self(idx, kv_caches=kv_caches, active_indices=active_indices, is_thinking=is_thinking)
            # pluck the logits at the final step and scale by desired temperature
            logits_step = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits_step, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Update thinking state based on the token that will be passed in next
            for b_idx in range(idx.size(0)):
                if idx_next[b_idx] == 256:
                    is_thinking[b_idx] = True
                elif idx_next[b_idx] == 257:
                    is_thinking[b_idx] = False
            
            # append sampled index to the running sequence and continue
            full_idx = torch.cat((full_idx, idx_next), dim=1)
            idx = idx_next

        return full_idx