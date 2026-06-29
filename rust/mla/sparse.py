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

    def _sparse_forward(self, q, k, v):
        B, NH, S, HD = q.shape
        NK = self.num_kv_groups; HPG = NH // NK
        BSZ = self.block_size
        K = min(self.num_selected_blocks, max(1, (S + BSZ - 1) // BSZ))
        NB = max(1, (S + BSZ - 1) // BSZ)

        pad = NB * BSZ - S
        if pad:
            k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))

        k_b = k[:, :NK].reshape(B, NK, NB, BSZ, HD)
        v_b = v[:, :NK].reshape(B, NK, NB, BSZ, HD)
        k_comp = k_b.mean(dim=3)
        q_g = q.reshape(B, NK, HPG, S, HD).mean(dim=2)

        scores = (q_g @ k_comp.transpose(-2, -1)) * (HD ** -0.5)
        if self.causal:
            q_pos = torch.arange(S, device=q.device)
            scores.masked_fill_(
                (q_pos[:, None] < (torch.arange(NB, device=q.device) * BSZ)[None, :])
                .unsqueeze(0).unsqueeze(0), float('-inf'))

        _, topk = scores.topk(K, dim=-1)  # (B, NK, S, K)

        out = torch.zeros_like(q)
        for blk in range(NB):
            mask = (topk == blk).any(dim=-1)
            if not mask.any():
                continue
            for g in range(NK):
                idx = mask[:, g]
                if not idx.any():
                    continue
                q_grp = q[:, g*HPG:(g+1)*HPG]  # (B, HPG, S, HD)
                kg = k_b[:, g, blk]  # (B, BSZ, HD)
                vg = v_b[:, g, blk]
                for h_off in range(HPG):
                    h = g * HPG + h_off
                    qh = q_grp[:, h_off]  # (B, S, HD)
                    n = idx[0].sum().item()
                    if n == 0:
                        continue
                    qm = qh[:, idx[0]].transpose(0, 1)  # (n, 1, HD)
                    km = kg.expand(n, -1, -1)  # (n, BSZ, HD)
                    vm = vg.expand(n, -1, -1)
                    o = F.scaled_dot_product_attention(qm, km, vm, is_causal=False)
                    out[0, h, idx[0]] = o.squeeze(1)
        return out[:, :, :S] if pad else out

    def forward(self, x, offset=0):
        q, k, v = self.qkv(x)
        q, k = self.rope(q, k, offset)

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if not self.training and q.shape[2] < 2048:
            return super().forward(x, offset)

        attn_output = self._sparse_forward(q, k, v)
        attn_output = attn_output.transpose(1, 2)
        return self.o_proj(attn_output)

    def forward_with_cache(self, x, offset, cache):
        return super().forward_with_cache(x, offset, cache)
