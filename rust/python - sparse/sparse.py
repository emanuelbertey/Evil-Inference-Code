import torch
import torch.nn.functional as F
from attention import Attention, repeat_kv


class SparseAttentionMio3(Attention):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim,
                 max_seq_len=2048, rope_base=10000.0, rope_scaling=1.0,
                 causal=True, dropout=0.0, attn_logit_cap=None, bias=False,
                 block_size=128, num_selected_blocks=16):
        super().__init__(d_model, num_heads, num_kv_groups, head_dim,
                         max_seq_len, rope_base, rope_scaling, causal,
                         dropout, attn_logit_cap, bias)
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks
        self.chunk_size = 64

    def _sparse_forward(self, q, k, v):
        B, NH, S, HD = q.shape
        NK = self.num_kv_groups; HPG = NH // NK
        BSZ = self.block_size
        K = min(self.num_selected_blocks, max(1, (S + BSZ - 1) // BSZ))
        CHUNK = self.chunk_size
        scale = 1.0 / (HD ** 0.5)
        NB = max(1, (S + BSZ - 1) // BSZ)

        pad = NB * BSZ - S
        if pad:
            k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))

        k_b = k[:, :NK].reshape(B, NK, NB, BSZ, HD)
        v_b = v[:, :NK].reshape(B, NK, NB, BSZ, HD)
        k_comp = k_b.mean(dim=3)

        out = torch.zeros_like(q)
        for start in range(0, S, CHUNK):
            end = min(start + CHUNK, S)
            q_chunk = q[:, :, start:end]
            C = end - start

            q_g = q_chunk.reshape(B, NK, HPG, C, HD).mean(dim=2)
            scores = (q_g @ k_comp.transpose(-2, -1)) * scale
            if self.causal:
                q_pos = torch.arange(start, start + C, device=q.device)
                scores.masked_fill_(
                    (q_pos[:, None] < (torch.arange(NB, device=q.device) * BSZ)[None, :])
                    .unsqueeze(0).unsqueeze(0), float('-inf'))

            _, topk = scores.topk(K, dim=-1)

            idx_b = torch.arange(B, device=q.device)[:, None, None, None]
            k_sel = k_b[idx_b, torch.arange(NK, device=q.device)[None, :, None, None], topk]
            v_sel = v_b[idx_b, torch.arange(NK, device=q.device)[None, :, None, None], topk]

            k_sel = k_sel.repeat_interleave(HPG, dim=1)
            v_sel = v_sel.repeat_interleave(HPG, dim=1)

            q_flat = q_chunk.reshape(B * NH * C, 1, HD)
            k_flat = k_sel.reshape(B * NH * C, K * BSZ, HD)
            v_flat = v_sel.reshape(B * NH * C, K * BSZ, HD)

            out_chunk = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, is_causal=False)
            out[:, :, start:end] = out_chunk.reshape(B, NH, C, HD)

            del q_chunk, q_g, scores, topk, k_sel, v_sel, q_flat, k_flat, v_flat, out_chunk

        return out[:, :, :S] if pad else out

    def forward(self, x, offset=0):
        q, k, v = self.qkv(x)
        q, k = self.rope(q, k, offset)

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if not self.training and q.shape[2] < 2048 and False:
            return super().forward(x, offset)

        attn_output = self._sparse_forward(q, k, v)
        attn_output = attn_output.transpose(1, 2)
        return self.o_proj(attn_output)

    def forward_with_cache(self, x, offset, cache):
        return super().forward_with_cache(x, offset, cache)
