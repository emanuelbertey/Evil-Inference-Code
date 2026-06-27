"""SparseAttention propio: sliding window + block-sparse global.

Memoria O(n * W + n * block_size * K) vs O(n^2) de GQA.
Sliding window vía SDPA chunked, block-sparse vía mean-pool + top-k con grupos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class SparseAttentionMio(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, causal=True,
                 sliding_window=256, block_size=32, num_selected_blocks=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.sliding_window = sliding_window
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks
        self.scale = head_dim ** -0.5

        dim_q = num_heads * head_dim
        dim_kv = num_kv_groups * head_dim
        self.qkv_proj = nn.Linear(d_model, dim_q + 2 * dim_kv, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        NH, NK, HD = self.num_heads, self.num_kv_groups, self.head_dim

        qkv = self.qkv_proj(x)
        q = qkv[:, :, :NH * HD].view(B, S, NH, HD)
        k = qkv[:, :, NH * HD:NH * HD + NK * HD].view(B, S, NK, HD)
        v = qkv[:, :, NH * HD + NK * HD:].view(B, S, NK, HD)

        k_e = _repeat_kv(k, NH, NK).transpose(1, 2)
        v_e = _repeat_kv(v, NH, NK).transpose(1, 2)
        q_t = q.transpose(1, 2)

        local = self._sliding(q_t, k_e, v_e, S)
        global_ = self._block_global(q_t, k_e, v_e, B, NH, NK, S, HD)

        out = (local + global_).transpose(1, 2).contiguous().view(B, S, NH * HD)
        return self.o_proj(out)

    def _sliding(self, q, k, v, S):
        W = self.sliding_window
        if S <= W:
            return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = torch.zeros_like(q)
        for s in range(0, S, W):
            e = min(s + W, S)
            ks = max(0, s - W)
            out[:, :, s:e] = F.scaled_dot_product_attention(
                q[:, :, s:e], k[:, :, ks:e], v[:, :, ks:e], is_causal=(ks == 0))
        return out

    def _block_global(self, q, k, v, B, NH, NK, S, HD):
        BS = self.block_size
        K = self.num_selected_blocks
        if S <= BS * 2:
            return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        NB = (S + BS - 1) // BS
        pad = NB * BS - S
        if pad > 0:
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))

        k_blocks = k.view(B, NH, NB, BS, HD)
        v_blocks = v.view(B, NH, NB, BS, HD)
        k_comp = k_blocks.mean(dim=3)
        v_comp = v_blocks.mean(dim=3)

        scores = torch.matmul(q, k_comp.transpose(-2, -1)) * self.scale
        if self.causal:
            last = torch.arange(BS, NB * BS + 1, BS, device=q.device).clamp(max=S)
            mask = torch.arange(S, device=q.device).unsqueeze(1) < last.unsqueeze(0)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        K_act = min(K, NB)
        _, topk = scores.topk(K_act, dim=-1)
        topk = topk.sort(dim=-1).values

        out = torch.zeros_like(q)
        for g in range(0, S, 32):
            ge = min(g + 32, S)
            idx = topk[:, :, g:ge].unsqueeze(-1).unsqueeze(-1)
            k_sel = k_blocks.unsqueeze(2).expand(-1, -1, ge - g, -1, -1, -1)
            k_sel = k_sel.gather(3, idx.expand(-1, -1, ge - g, -1, BS, HD))
            v_sel = v_blocks.unsqueeze(2).expand(-1, -1, ge - g, -1, -1, -1)
            v_sel = v_sel.gather(3, idx.expand(-1, -1, ge - g, -1, BS, HD))
            k_sel = k_sel.view(B, NH, ge - g, K_act * BS, HD)
            v_sel = v_sel.view(B, NH, ge - g, K_act * BS, HD)
            q_g = q[:, :, g:ge].unsqueeze(3)
            s = (q_g @ k_sel.transpose(-2, -1)).squeeze(3) * self.scale
            a = F.softmax(s, dim=-1)
            out[:, :, g:ge] = (a.unsqueeze(-2) @ v_sel).squeeze(3)

        return out[:, :, :S] if pad > 0 else out
