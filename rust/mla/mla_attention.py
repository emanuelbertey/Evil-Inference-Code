"""Multi-head Latent Attention (MLA) with GQA.

Based on DeepSeek MLA adapted for GQA.
- Latent KV compression, shared latent across KV groups
- RoPE on separate rotation dimension
- Latent cache (C_KV, K_rotate_raw) en vez de K/V expandido
- Dense attention (no sparse)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE, apply_rope_partial
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

        self.W_down = nn.Linear(d_model, d_c1 + d_c + d_rotate, bias=bias)
        self.W_up_q = nn.Linear(d_c1, num_heads * (head_dim + d_rotate), bias=bias)
        self.W_up_kv = nn.Linear(d_c, 2 * num_kv_groups * head_dim, bias=bias)

    def forward(self, x):
        B, S, _ = x.shape
        down = self.W_down(x)
        C_Q, C_KV, K_rotate = down.split([self.d_c1, self.d_c, self.d_rotate], dim=-1)

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

    def _attention_scores(self, Q, K, V, q_len, kv_len, causal_mask):
        q = Q.transpose(1, 2)
        k = K.transpose(1, 2)
        v = V.transpose(1, 2)
        scale = math.sqrt(self.qkv.qk_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap
        if self.causal and causal_mask:
            mask = torch.triu(torch.full((q_len, kv_len), float("-inf"), device=scores.device), diagonal=kv_len - q_len + 1).unsqueeze(0).unsqueeze(0)
            scores = scores + mask
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        return torch.matmul(attn_w, v).transpose(1, 2)

    def forward_with_cache(self, x, offset, cache):
        B, S_new, _ = x.shape
        down = self.qkv.W_down(x)
        C_Q_new, C_KV_new, K_rot_raw = down.split([self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        q_up = self.qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.qkv.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, self.num_heads, self.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, self.num_heads, self.qkv.d_rotate)
        Q_rot = self.rope.apply_to_single(Q_rot_raw, offset=offset)

        if cache is not None:
            C_KV_full = torch.cat([cache[0], C_KV_new], dim=1)
            K_rot_full = torch.cat([cache[1], K_rot_raw], dim=1)
        else:
            C_KV_full = C_KV_new
            K_rot_full = K_rot_raw
        S_full = C_KV_full.shape[1]

        kv_up = self.qkv.W_up_kv(C_KV_full)
        K_state, V_state = kv_up.chunk(2, dim=-1)
        K_state = K_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)
        V_state = V_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)

        K_rot = self.rope.apply_to_single(K_rot_full.unsqueeze(2), offset=0)
        K_rot_exp = K_rot.expand(-1, -1, self.num_kv_groups, -1)

        K = torch.cat([K_state, K_rot_exp], dim=-1)
        k = repeat_kv(K, self.num_heads, self.num_kv_groups)
        v = repeat_kv(V_state, self.num_heads, self.num_kv_groups)

        Q = torch.cat([Q_state, Q_rot], dim=-1)
        attn_out = self._attention_scores(Q, k, v, S_new, S_full, S_new > 1)
        return self.o_proj(attn_out), (C_KV_full, K_rot_full)

    def forward_with_cache_partial(self, x, offset, cache, rotary_pct):
        B, S_new, _ = x.shape
        down = self.qkv.W_down(x)
        C_Q_new, C_KV_new, K_rot_raw = down.split([self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        q_up = self.qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.qkv.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, self.num_heads, self.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, self.num_heads, self.qkv.d_rotate)

        Q_rot, _ = apply_rope_partial(Q_rot_raw, Q_rot_raw, offset, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.rope.head_dim, self.rope.max_seq_len)

        if cache is not None:
            C_KV_full = torch.cat([cache[0], C_KV_new], dim=1)
            K_rot_full = torch.cat([cache[1], K_rot_raw], dim=1)
        else:
            C_KV_full = C_KV_new
            K_rot_full = K_rot_raw
        S_full = C_KV_full.shape[1]

        kv_up = self.qkv.W_up_kv(C_KV_full)
        K_state, V_state = kv_up.chunk(2, dim=-1)
        K_state = K_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)
        V_state = V_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)

        _, K_rot = apply_rope_partial(K_rot_full.unsqueeze(2), K_rot_full.unsqueeze(2), 0, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.rope.head_dim, self.rope.max_seq_len)
        K_rot_exp = K_rot.expand(-1, -1, self.num_kv_groups, -1)

        K = torch.cat([K_state, K_rot_exp], dim=-1)
        k = repeat_kv(K, self.num_heads, self.num_kv_groups)
        v = repeat_kv(V_state, self.num_heads, self.num_kv_groups)

        Q = torch.cat([Q_state, Q_rot], dim=-1)
        attn_out = self._attention_scores(Q, k, v, S_new, S_full, S_new > 1)
        return self.o_proj(attn_out), (C_KV_full, K_rot_full)
