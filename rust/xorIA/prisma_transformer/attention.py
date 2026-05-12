import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import RMSNorm, make_linear, apply_rope
from config import PrismaConfig

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, n_kv_heads, S, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, S, head_dim)
        .reshape(B, n_kv_heads * n_rep, S, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, config: PrismaConfig, freqs_cis: torch.Tensor, lazy=True):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads_kv
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = make_linear(config.dim, config.n_heads * self.head_dim, bias=config.bias, quant_mode=config.quant_mode, lazy=lazy)
        self.wk = make_linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.bias, quant_mode=config.quant_mode, lazy=lazy)
        self.wv = make_linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.bias, quant_mode=config.quant_mode, lazy=lazy)
        self.wo = make_linear(config.n_heads * self.head_dim, config.dim, bias=config.bias, quant_mode=config.quant_mode, lazy=lazy)

        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.attn_sub_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.use_attn_sub_norm = False
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x, kv_cache=None, offset=0):
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, self.freqs_cis, offset=offset)
        k = apply_rope(k, self.freqs_cis, offset=offset)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k.detach(), v.detach())
        k_attn = repeat_kv(k, self.n_rep)
        v_attn = repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if S > 1:
            kv_len = k_attn.size(2)
            q_pos = torch.arange(offset, offset + S, device=x.device).view(S, 1)
            k_pos = torch.arange(kv_len, device=x.device).view(1, kv_len)
            scores = scores.masked_fill(k_pos > q_pos, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores.float(), dim=-1).type_as(x)
        out = torch.matmul(attn, v_attn).transpose(1, 2).contiguous().view(B, S, -1)
        if self.use_attn_sub_norm:
            out = self.attn_sub_norm(out)
        return self.wo(out), new_cache
