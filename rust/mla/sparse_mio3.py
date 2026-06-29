"""SparseAttention v3 — MSA port, softmax global, unificado.

Por chunk: gather UNICO (B, NH, C, K, BS, HD) → reshape
→ una SDPA con batch = B*NH*C queries.

Sin bucles por cabeza ni por bloque.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class SparseAttentionMio3(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, causal=True,
                 block_size=128, num_selected_blocks=16):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks
        self.chunk_size = 8

        dim_q = num_heads * head_dim
        dim_kv = num_kv_groups * head_dim
        self.qkv_proj = nn.Linear(d_model, dim_q + 2 * dim_kv, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def _chunk_forward(self, q_chunk, k_b, v_b, start, K_act, NB, BSZ, CHUNK):
        B, NH, C, HD = q_chunk.shape

        k_comp = k_b.mean(dim=3)
        scores = (q_chunk @ k_comp.transpose(-2, -1)) / (HD ** 0.5)
        if self.causal:
            q_pos = torch.arange(start, start + C, device=q_chunk.device)
            scores.masked_fill_(
                (q_pos[:, None] < (torch.arange(NB, device=q_chunk.device) * BSZ)[None, :])
                .unsqueeze(0).unsqueeze(0),
                float('-inf')
            )

        _, topk = scores.topk(K_act, dim=-1)

        # Gather único: (B, NH, C, K, BS, HD)
        idx_b = torch.arange(B, device=q_chunk.device)[:, None, None, None]       # (B, 1, 1, 1)
        idx_h = torch.arange(NH, device=q_chunk.device)[None, :, None, None]      # (1, NH, 1, 1)
        k_sel = k_b[idx_b, idx_h, topk]   # (B, NH, C, K, BS, HD)
        v_sel = v_b[idx_b, idx_h, topk]

        # Una SDPA: flat (B*NH*C, 1, HD) × (B*NH*C, K*BS, HD)
        q_flat = q_chunk.reshape(B * NH * C, 1, HD)
        k_flat = k_sel.reshape(B * NH * C, K_act * BSZ, HD)
        v_flat = v_sel.reshape(B * NH * C, K_act * BSZ, HD)

        out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, is_causal=False)
        return out_flat.reshape(B, NH, C, HD)

    def forward(self, x):
        B, S, _ = x.shape
        NH, NK, HD, BSZ, K = self.num_heads, self.num_kv_groups, self.head_dim, self.block_size, self.num_selected_blocks
        CHUNK = self.chunk_size

        qkv = self.qkv_proj(x)
        q = qkv[:, :, :NH * HD].reshape(B, S, NH, HD).permute(0, 2, 1, 3)
        k = qkv[:, :, NH * HD:NH * HD + NK * HD].reshape(B, S, NK, HD)
        v = qkv[:, :, NH * HD + NK * HD:].reshape(B, S, NK, HD)

        k = _repeat_kv(k, NH, NK).permute(0, 2, 1, 3)
        v = _repeat_kv(v, NH, NK).permute(0, 2, 1, 3)

        NB = max(1, (S + BSZ - 1) // BSZ)
        pad = NB * BSZ - S
        if pad > 0:
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        k_b = k.reshape(B, NH, NB, BSZ, HD)
        v_b = v.reshape(B, NH, NB, BSZ, HD)
        K_act = min(K, NB)

        out = torch.zeros_like(q)
        for start in range(0, S, CHUNK):
            end = min(start + CHUNK, S)
            q_chunk = q[:, :, start:end]

            if self.training and start > 0:
                contrib = torch.utils.checkpoint.checkpoint(
                    self._chunk_forward, q_chunk, k_b, v_b,
                    torch.tensor(start), K_act, NB, BSZ, CHUNK,
                    use_reentrant=False
                )
            else:
                contrib = self._chunk_forward(q_chunk, k_b, v_b, start, K_act, NB, BSZ, CHUNK)

            out[:, :, start:end] = contrib

        if pad > 0:
            out = out[:, :, :S]

        out = out.permute(0, 2, 1, 3).reshape(B, S, NH * HD)
        return self.o_proj(out)
