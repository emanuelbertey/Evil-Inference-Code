"""Multi-head Latent Attention (MLA) with GQA.

Based on DeepSeek MLA adapted for GQA.
- Latent KV compression with per-group latents
- RoPE on separate rotation dimension
- Dense attention (no sparse)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE, apply_rope_partial
from cache_kv import KVCache
from attention import repeat_kv


class QKVProjectionMLA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, d_c, d_c1, d_rotate, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rotate = d_rotate
        self.qk_dim = head_dim + d_rotate

        self.W_down = nn.Linear(d_model, d_c1 + num_kv_groups * d_c + d_rotate, bias=bias)
        self.W_up_q = nn.Linear(d_c1, d_model + num_heads * d_rotate, bias=bias)
        self.W_up_kv = nn.Linear(num_kv_groups * d_c, 2 * num_kv_groups * head_dim, bias=bias)

    def forward(self, x):
        B, S, _ = x.shape
        down = self.W_down(x)
        C_Q, C_KV, K_rotate = down.split([self.d_c1, self.num_kv_groups * self.d_c, self.d_rotate], dim=-1)

        q_up = self.W_up_q(C_Q)
        Q_state, Q_rotate = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S, self.num_heads, self.head_dim)
        Q_rotate = Q_rotate.reshape(B, S, self.num_heads, self.d_rotate)

        kv_up = self.W_up_kv(C_KV)
        K, V = kv_up.chunk(2, dim=-1)
        K = K.reshape(B, S, self.num_kv_groups, self.head_dim)
        V = V.reshape(B, S, self.num_kv_groups, self.head_dim)
        K_rotate = K_rotate.reshape(B, S, 1, self.d_rotate)

        return Q_state, Q_rotate, K, V, K_rotate


class OutputProjectionMLA(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, bias=False):
        super().__init__()
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)

    def forward(self, x):
        B, S, NH, QK = x.shape
        return self.o_proj(x.reshape(B, S, NH * QK))


class MultiHeadLatentAttentionGQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups=0, head_dim=None,
                 max_seq_len=2048, rope_base=10000.0, rope_scaling=1.0,
                 causal=True, dropout=0.0, attn_logit_cap=None, bias=False,
                 d_c=None, d_c1=None, d_rotate=None, block_size=128):
        super().__init__()
        if num_kv_groups == 0: num_kv_groups = num_heads
        if head_dim is None: head_dim = d_model // num_heads
        if d_c is None: d_c = max(32, d_model // 6)
        if d_c1 is None: d_c1 = max(32, d_model // 6)
        if d_rotate is None: d_rotate = max(16, d_model // 12)

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.block_size = block_size

        self.qkv = QKVProjectionMLA(d_model, num_heads, num_kv_groups, head_dim,
                                     d_c, d_c1, d_rotate, bias)
        self.o_proj = OutputProjectionMLA(d_model, num_heads, head_dim, bias)
        self.rope = RoPE(head_dim=d_rotate, max_seq_len=max_seq_len, base=rope_base, scaling_factor=rope_scaling)
        self.rope.head_dim = d_rotate
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, offset=0):
        Q_state, Q_rotate, K, V, K_rotate = self.qkv(x)
        Q_rotate, K_rotate = self.rope(Q_rotate, K_rotate, offset)

        Q = torch.cat([Q_state, Q_rotate], dim=-1)
        K_rot_exp = K_rotate.expand(-1, -1, self.num_kv_groups, -1)
        K = torch.cat([K, K_rot_exp], dim=-1)

        k = repeat_kv(K, self.num_heads, self.num_kv_groups)
        v = repeat_kv(V, self.num_heads, self.num_kv_groups)

        q = Q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scale = math.sqrt(self.qkv.qk_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        seq_len = q.shape[2]
        if self.causal and seq_len > 1:
            scores = scores + torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1).unsqueeze(0).unsqueeze(0)

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        attn_out = torch.matmul(attn_w, v).transpose(1, 2)
        return self.o_proj(attn_out)

    def forward_with_cache(self, x, offset, cache):
        Q_state, Q_rotate, K_new, V_new, K_rot_new = self.qkv(x)
        Q_rotate, K_rot_new = self.rope(Q_rotate, K_rot_new, offset)

        K_rot_exp = K_rot_new.expand(-1, -1, self.num_kv_groups, -1)
        K_full_new = torch.cat([K_new, K_rot_exp], dim=-1)

        k_full = torch.cat([cache.cached_k, K_full_new], dim=1) if cache else K_full_new
        v_full = torch.cat([cache.cached_v, V_new], dim=1) if cache else V_new
        new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())

        Q = torch.cat([Q_state, Q_rotate], dim=-1)
        k_exp = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_exp = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = Q.transpose(1, 2); k = k_exp.transpose(1, 2); v = v_exp.transpose(1, 2)
        scale = math.sqrt(self.qkv.qk_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        q_len, kv_len = q.shape[2], k.shape[2]
        if self.causal and q_len > 1:
            scores = scores + torch.triu(torch.full((q_len, kv_len), float("-inf"), device=scores.device), diagonal=kv_len - q_len + 1).unsqueeze(0).unsqueeze(0)

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        attn_out = torch.matmul(attn_w, v).transpose(1, 2)
        return self.o_proj(attn_out), new_cache

    def forward_with_cache_partial(self, x, offset, cache, rotary_pct):
        Q_state, Q_rotate, K_new, V_new, K_rot_new = self.qkv(x)
        Q_rotate, K_rot_new = apply_rope_partial(Q_rotate, K_rot_new, offset, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.rope.head_dim, self.rope.max_seq_len)

        K_rot_exp = K_rot_new.expand(-1, -1, self.num_kv_groups, -1)
        K_full_new = torch.cat([K_new, K_rot_exp], dim=-1)

        k_full = torch.cat([cache.cached_k, K_full_new], dim=1) if cache else K_full_new
        v_full = torch.cat([cache.cached_v, V_new], dim=1) if cache else V_new
        new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())

        Q = torch.cat([Q_state, Q_rotate], dim=-1)
        k_exp = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_exp = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = Q.transpose(1, 2); k = k_exp.transpose(1, 2); v = v_exp.transpose(1, 2)
        scale = math.sqrt(self.qkv.qk_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        q_len, kv_len = q.shape[2], k.shape[2]
        if self.causal and q_len > 1:
            scores = scores + torch.triu(torch.full((q_len, kv_len), float("-inf"), device=scores.device), diagonal=kv_len - q_len + 1).unsqueeze(0).unsqueeze(0)

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        attn_out = torch.matmul(attn_w, v).transpose(1, 2)
        return self.o_proj(attn_out), new_cache
