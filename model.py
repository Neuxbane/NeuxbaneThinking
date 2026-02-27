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
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(config.block_size).float()
        freqs = torch.outer(t, inv_freq) # (block_size, head_dim // 2)
        self.register_buffer("cos", freqs.cos().view(1, 1, config.block_size, self.head_dim // 2), persistent=False)
        self.register_buffer("sin", freqs.sin().view(1, 1, config.block_size, self.head_dim // 2), persistent=False)

        # flash attention make GPU go brrr but for simplicity we use manual mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size), persistent=False)

    def _apply_rope(self, x, start_pos=0):
        B, nh, T, hs = x.size()
        # Clamp start_pos and T to ensure we stay within the precomputed cos/sin buffers
        sp = max(0, min(start_pos, self.config.block_size - 1))
        # Ensure we don't slice past the buffer even if the input is longer than block_size
        max_t = min(T, self.config.block_size - sp)
        
        cos = self.cos[:, :, sp:sp+max_t, :]
        sin = self.sin[:, :, sp:sp+max_t, :]
        
        x1 = x[..., :max_t, 0::2]
        x2 = x[..., :max_t, 1::2]
        
        out = x.clone()
        out[..., :max_t, 0::2] = x1 * cos - x2 * sin
        out[..., :max_t, 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, x, kv_cache=None):
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

        start_pos = 0 # Default starting position
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            current_cache_len = prev_k.shape[2]
            
            # Sliding Window with Attention Sinks (StreamingLLM style):
            # We preserve the first few tokens (Sink) to maintain global attention anchors.
            # We ensure the total KV cache length never exceeds config.block_size.
            if current_cache_len + T > self.config.block_size:
                sink_size = 4
                # T_effective is the amount of NEW tokens we can fit alongside sinks
                T_effective = min(T, self.config.block_size - sink_size)
                # body_len is the amount of PREVIOUS context we can preserve
                body_len = max(0, self.config.block_size - T_effective - sink_size)
                
                # Slice previous cache
                sink_k = prev_k[:, :, :sink_size, :]
                sink_v = prev_v[:, :, :sink_size, :]
                
                if body_len > 0:
                    body_k = prev_k[:, :, -body_len:, :]
                    body_v = prev_v[:, :, -body_len:, :]
                    prev_k = torch.cat([sink_k, body_k], dim=2)
                    prev_v = torch.cat([sink_v, body_v], dim=2)
                else:
                    prev_k = sink_k
                    prev_v = sink_v
                
                # Truncate new tokens if they are too long to fit with sinks
                if T > T_effective:
                    k = k[:, :, -T_effective:, :]
                    v = v[:, :, -T_effective:, :]
                    q = q[:, :, -T_effective:, :]
                    T = T_effective
                
                # Position new tokens at the end of the sliding window
                start_pos = self.config.block_size - T
            else:
                start_pos = current_cache_len
            
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        new_kv_cache = (k, v)
        
        # repeat KV heads if n_kv_head < n_head
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            # (B, n_kv_head, T_total, head_dim) -> (B, n_kv_head, n_rep, T_total, head_dim) -> (B, n_head, T_total, head_dim)
            k = k[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, k.size(2), self.head_dim).reshape(B, self.n_head, k.size(2), self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, v.size(2), self.head_dim).reshape(B, self.n_head, v.size(2), self.head_dim)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T_total) -> (B, nh, T, T_total)
        T_total = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        if T > 1:
            # Note: Masking for sink tokens is naturally handled because sink tokens 
            # are at the beginning of the sequence and are never "ahead" of anything.
            mask = self.bias[:,:,start_pos:start_pos+T,:T_total]
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T_total) x (B, nh, T_total, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y, new_kv_cache

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, q, k, v):
        B, Tq, C = q.size()
        B, Tk, Ck = k.size()
        
        q = self.q_proj(q).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.c_proj(y)

class LinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.eps = 1e-6

    def forward(self, q, k, v):
        B, Tq, C = q.size()
        B, Tk, Ck = k.size()
        
        q = self.q_proj(q).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.c_proj(y)

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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        
        if config.n_scratchpad > 0:
            # Routing mechanism to pick which scratchpads to use
            # Input comes from the model's own state at the current position's beginning
            self.router = nn.Linear(config.n_embd, config.num_scratchpads, bias=False)
            
            # Tokens query the scratchpad to "read" from memory
            self.ln_read_tok = RMSNorm(config.n_embd)
            self.ln_read_sp  = RMSNorm(config.n_embd)
            self.read_attn = CrossAttention(config)
            
            # Scratchpad evolves internally (Self-Attention within memory)
            self.ln_write_sp  = RMSNorm(config.n_embd)
            self.ln_write_sp_kv = RMSNorm(config.n_embd)
            self.write_attn = CrossAttention(config)

            # Gated memory update logic (replaces simple residual)
            self.sp_gate = nn.Linear(2 * config.n_embd, config.n_embd)
            self.sp_update_proj = nn.Linear(config.n_embd, config.n_embd)
            self.ln_sp_update = RMSNorm(config.n_embd)

        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, scratchpad=None, kv_cache=None):
        # 1. Self-Attention on tokens
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        
        if scratchpad is not None:
            B, T, C = x.shape
            num_pages = self.config.num_scratchpads
            n_sp = self.config.n_scratchpad
            
            # Use current token for routing (consistent across training/inference)
            x_router_input = self.ln_1(x) 
                
            router_logits = self.router(x_router_input) # (B, T, num_scratchpads)
            router_probs = F.softmax(router_logits, dim=-1)
            top_probs, top_indices = torch.topk(router_probs, 2, dim=-1) # (B, T, 2)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            
            # Gather page-level summaries for tokens (Softmax compatibility)
            # To keep memory bounded, we attend to each selected page individually 
            # and then combine, rather than materializing a token-slots cross-product.
            x_tok = self.ln_read_tok(x)
            sp_read = self.ln_read_sp(scratchpad)
            
            # Optimized Cross-Attention: Compute scores for ALL pages and then mask,
            # avoiding multi-gigabyte intermediate tensors (B, T, 2, n_sp, nh, hs).
            q_read = self.read_attn.q_proj(x_tok).view(B, T, self.config.n_head, self.read_attn.head_dim).transpose(1, 2)
            k_read = self.read_attn.k_proj(sp_read).view(B, num_pages, n_sp, self.config.n_head, self.read_attn.head_dim).permute(0, 3, 1, 2, 4)
            v_read = self.read_attn.v_proj(sp_read).view(B, num_pages, n_sp, self.config.n_head, self.read_attn.head_dim).permute(0, 3, 1, 2, 4)
            
            # Scores for all slots: (B, nh, T, num_pages, n_sp)
            all_scores = torch.einsum('bnth,bnpsh->bntps', q_read, k_read) * (1.0 / math.sqrt(self.read_attn.head_dim))
            
            # Mask to only allow attention to the 2 selected pages per token
            router_mask = torch.zeros(B, T, num_pages, device=x.device)
            router_mask.scatter_(2, top_indices, 1.0)
            all_scores = all_scores.masked_fill(router_mask.view(B, 1, T, num_pages, 1) == 0, float('-inf'))
            
            # Softmax and aggregate
            attn_probs = F.softmax(all_scores.view(B, self.config.n_head, T, num_pages * n_sp), dim=-1)
            attn_probs = attn_probs.view(B, self.config.n_head, T, num_pages, n_sp)
            
            read_out = torch.einsum('bntps,bnpsh->bnth', attn_probs, v_read)
            read_out = read_out.transpose(1, 2).reshape(B, T, C)
            
            # Stability Gate: Add memory to context via a learned projection
            x = x + self.read_attn.c_proj(read_out)

            # --- 3. Memory WRITING (Internal Evolution) ---
            # Self-attention within pages (B*num_pages, n_sp, C)
            sp_q = self.ln_write_sp(scratchpad).view(B * num_pages, n_sp, C)
            sp_kv = self.ln_write_sp_kv(scratchpad).view(B * num_pages, n_sp, C)
            
            # Use standard CrossAttention for self-evolution to match Softmax weights
            raw_update = self.write_attn(sp_q, sp_kv, sp_kv).view(B, num_pages, n_sp, C)
            
            # Gated Update
            gate = torch.sigmoid(self.sp_gate(torch.cat([scratchpad, raw_update], dim=-1)))
            update_delta = gate * torch.tanh(self.sp_update_proj(raw_update))
            
            # Apply update priority based on router hits to tie routing to maintenance gradients
            # This ensures only 'useful' pages learn to evolve effectively.
            update_mask = torch.zeros(B, num_pages, device=x.device)
            # Add weights of tokens that selected each page
            update_mask.scatter_add_(1, top_indices.view(B, -1), top_probs.view(B, -1))
            
            # Normalize update speed: We update by (hits / block_size) unit per forward pass.
            # This synchronizes the "evolution speed" between the training block (batch of tokens)
            # and inference (single token). It keeps the memory state from drifting too fast during generation.
            update_mask = update_mask / self.config.block_size
            
            scratchpad_new = scratchpad + update_delta * update_mask.view(B, num_pages, 1, 1)
            scratchpad = self.ln_sp_update(scratchpad_new)

        # 4. MLP on tokens
        x = x + self.mlp(self.ln_2(x))
        return x, scratchpad, new_kv_cache

class TransformerConfig:
    def __init__(self, vocab_size=256, block_size=512, n_layer=6, n_head=6, n_kv_head=None, n_embd=384, n_scratchpad=512, num_scratchpads=4, rope_theta=10000.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_embd = n_embd
        self.n_scratchpad = n_scratchpad
        self.num_scratchpads = num_scratchpads
        self.rope_theta = rope_theta

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.n_scratchpad > 0:
            # Memory initialization state (per page)
            self.scratchpad_init = nn.Parameter(torch.randn(config.num_scratchpads, config.n_scratchpad, config.n_embd) * 0.02)
            # Learnable positional embeddings for memory slots to give them distinct indices/identities within a page
            self.scratchpad_pos = nn.Parameter(torch.randn(config.n_scratchpad, config.n_embd) * 0.02)
        else:
            self.scratchpad_init = None
            self.scratchpad_pos = None

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        # subtract tied weight
        n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None, scratchpad=None, kv_caches=None):
        device = idx.device
        b, t = idx.size()
        n_sp = self.config.n_scratchpad
        
        if kv_caches is None:
            # Prefill / Initial forward pass
            if t > self.config.block_size:
                # StreamingLLM-friendly truncation: Keep first 4 (sinks) and last N-4
                sink_size = 4
                idx = torch.cat([idx[:, :sink_size], idx[:, -(self.config.block_size - sink_size):]], dim=1)
                t = self.config.block_size
                if targets is not None:
                    targets = torch.cat([targets[:, :sink_size], targets[:, -(self.config.block_size - sink_size):]], dim=1)
            
            x = self.transformer.wte(idx) 
            if n_sp > 0:
                if scratchpad is None:
                    # Contextualize: provide initial model direction from the first few tokens 
                    # without allowing any future context during parallel training
                    topic_seed = x[:, 0:1, :].mean(dim=1, keepdim=True).unsqueeze(1) # (B, 1, 1, C)
                    # Initialize with both base state and slot-specific positional info
                    # Result shape: (B, num_scratchpads, n_scratchpad, n_embd)
                    scratchpad = (self.scratchpad_init + self.scratchpad_pos).unsqueeze(0).expand(b, -1, -1, -1)
                    scratchpad = scratchpad + topic_seed # Broadcast context to internal brain
            kv_caches = [None] * len(self.transformer.h)
        else:
            # Incremental generation pass
            x = self.transformer.wte(idx)
            if n_sp > 0 and scratchpad is None:
                topic_seed = x[:, 0:1, :].mean(dim=1, keepdim=True).unsqueeze(1)
                scratchpad = (self.scratchpad_init + self.scratchpad_pos).unsqueeze(0).expand(b, -1, -1, -1)
                scratchpad = scratchpad + topic_seed

        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            x, scratchpad, cache = block(x, scratchpad=scratchpad, kv_cache=kv_caches[i])
            new_kv_caches.append(cache)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return logits, loss, scratchpad, new_kv_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        full_idx = idx
        scratchpad = None
        kv_caches = None

        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _, scratchpad, kv_caches = self(idx, scratchpad=scratchpad, kv_caches=kv_caches)
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
            # append sampled index to the running sequence and continue
            full_idx = torch.cat((full_idx, idx_next), dim=1)
            idx = idx_next

        return full_idx
