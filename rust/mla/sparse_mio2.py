"""SparseAttention v2: sliding window + compressed blocks, todo vía SDPA con máscara.

Sin expand, sin gather, sin top-k por query. Máscara fija que permite:
- Sliding window: últimos W tokens
- Compressed: cada posición atiende a todos los bloques comprimidos (mean-pool)

SDPA con mask NO materializa scores completos (usa backend eficiente).
Memoria ~ O(S * (W + S/BS)) en vez de O(S²).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class SparseAttentionMio2(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, causal=True,
                 sliding_window=256, block_size=32):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.sliding_window = sliding_window
        self.block_size = block_size
        self.scale = head_dim ** -0.5

        dim_q = num_heads * head_dim
        dim_kv = num_kv_groups * head_dim
        self.qkv_proj = nn.Linear(d_model, dim_q + 2 * dim_kv, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        NH, NK, HD, W, BS = self.num_heads, self.num_kv_groups, self.head_dim, self.sliding_window, self.block_size

        qkv = self.qkv_proj(x)
        q = qkv[:, :, :NH * HD].contiguous().view(B, S, NH, HD)
        k = qkv[:, :, NH * HD:NH * HD + NK * HD].contiguous().view(B, S, NK, HD)
        v = qkv[:, :, NH * HD + NK * HD:].contiguous().view(B, S, NK, HD)

        # 1. Compressed global block summaries
        NB = max(1, (S + BS - 1) // BS)
        pad = NB * BS - S
        if pad > 0:
            k_blocks = F.pad(k, (0, 0, 0, 0, 0, pad)).view(B, -1, NK, NB, BS, HD)
            v_blocks = F.pad(v, (0, 0, 0, 0, 0, pad)).view(B, -1, NK, NB, BS, HD)
        else:
            k_blocks = k.view(B, -1, NK, NB, BS, HD)
            v_blocks = v.view(B, -1, NK, NB, BS, HD)

        k_comp = k_blocks.mean(dim=4)
        v_comp = v_blocks.mean(dim=4)

        # 2. Build combined KV: [compressed_blocks (NB), sliding_window (S)]
        k_comp_e = _repeat_kv(k_comp, NH, NK).view(B, 1, NH, NB, HD)
        v_comp_e = _repeat_kv(v_comp, NH, NK).view(B, 1, NH, NB, HD)
        k_e = _repeat_kv(k, NH, NK).view(B, S, NH, HD)
        v_e = _repeat_kv(v, NH, NK).view(B, S, NH, HD)

        q_t = q.permute(0, 2, 1, 3)  # (B, NH, S, HD)
        k_t = k_e.permute(0, 2, 1, 3)
        v_t = v_e.permute(0, 2, 1, 3)

        # 3. Global attention to compressed blocks
        k_comp_t = k_comp_e.squeeze(1).permute(0, 2, 1, 3).contiguous()
        v_comp_t = v_comp_e.squeeze(1).permute(0, 2, 1, 3).contiguous()
        global_out = F.scaled_dot_product_attention(q_t, k_comp_t, v_comp_t, is_causal=False)

        # 4. Local sliding window attention
        if S <= W:
            local_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=self.causal)
        else:
            local_out = torch.zeros_like(q_t)
            for s in range(0, S, W):
                e = min(s + W, S)
                ks = max(0, s - W)
                local_out[:, :, s:e] = F.scaled_dot_product_attention(
                    q_t[:, :, s:e], k_t[:, :, ks:e], v_t[:, :, ks:e], is_causal=(ks == 0))

        # 5. Combine
        out = (global_out + local_out).permute(0, 2, 1, 3).contiguous().view(B, S, NH * HD)
        return self.o_proj(out)
