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
        self.register_buffer("cos", freqs.cos().view(1, 1, config.block_size, self.head_dim // 2))
        self.register_buffer("sin", freqs.sin().view(1, 1, config.block_size, self.head_dim // 2))

        # flash attention make GPU go brrr but for simplicity we use manual mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

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
            
        # rotate queries and keys using their positions
        q = self._apply_rope(q, start_pos=start_pos)
        k = self._apply_rope(k, start_pos=start_pos)

        if kv_cache is not None:
            # kv_cache is (prev_k, prev_v)
            prev_k, prev_v = kv_cache
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
            # Tokens query the scratchpad to "read" from memory
            self.ln_read_tok = RMSNorm(config.n_embd)
            self.ln_read_sp  = RMSNorm(config.n_embd)
            self.read_attn = CrossAttention(config)
            
            # Scratchpad queries the tokens to "decide" what to write/update
            self.ln_write_sp  = RMSNorm(config.n_embd)
            self.ln_write_tok = RMSNorm(config.n_embd)
            self.write_attn = CrossAttention(config)

        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, scratchpad=None, kv_cache=None, start_pos_offset=0):
        # 1. Self-Attention on tokens
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, start_pos_offset=start_pos_offset)
        x = x + attn_out
        
        if scratchpad is not None:
            # 2. Tokens read from Scratchpad
            # Tokens are Q, Scratchpad is KV
            x = x + self.read_attn(self.ln_read_tok(x), self.ln_read_sp(scratchpad), self.ln_read_sp(scratchpad))
            
            # 3. Scratchpad writes from Tokens
            # Scratchpad is Q, Tokens are KV
            # Use Tanh to squash the update if desired, or keep as residual
            sp_update = self.write_attn(self.ln_write_sp(scratchpad), self.ln_write_tok(x), self.ln_write_tok(x))
            scratchpad = scratchpad + sp_update

        # 4. MLP on tokens
        x = x + self.mlp(self.ln_2(x))
        return x, scratchpad, new_kv_cache

class TransformerConfig:
    def __init__(self, vocab_size=256, block_size=512, n_layer=6, n_head=6, n_kv_head=None, n_embd=384, n_scratchpad=64, rope_theta=10000.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_embd = n_embd
        self.n_scratchpad = n_scratchpad
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
            self.scratchpad_init = nn.Parameter(torch.randn(config.n_scratchpad, config.n_embd) * 0.02)
        else:
            self.scratchpad_init = None

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
        start_pos_offset = n_sp if n_sp > 0 else 0
        
        if kv_caches is None:
            # Prefill / Initial forward pass
            if t + n_sp > self.config.block_size:
                t = self.config.block_size - n_sp
                idx = idx[:, -t:]
                if targets is not None:
                    targets = targets[:, -t:]

            x = self.transformer.wte(idx) 
            if n_sp > 0:
                if scratchpad is None:
                    scratchpad = self.scratchpad_init.unsqueeze(0).expand(b, -1, -1)
            kv_caches = [None] * len(self.transformer.h)
        else:
            # Incremental generation pass
            # 'idx' is the new token, 'scratchpad' is recycled from the previous step
            x = self.transformer.wte(idx)
            if n_sp > 0 and scratchpad is None:
                scratchpad = self.scratchpad_init.unsqueeze(0).expand(b, -1, -1)

        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            x, scratchpad, cache = block(x, scratchpad=scratchpad, kv_cache=kv_caches[i], start_pos_offset=start_pos_offset)
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
