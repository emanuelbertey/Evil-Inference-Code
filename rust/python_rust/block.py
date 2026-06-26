import torch
import torch.nn as nn

from .norm import RMSNorm
from .attention import GQAAttention
from .ffn import SwiGLUFFN


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_groups: int,
                 intermediate_dim: int, attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.0, residual_dropout: float = 0.0,
                 attn_logit_cap: float | None = None, causal: bool = True,
                 norm_eps: float = 1e-5):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attention = GQAAttention(
            d_model, num_heads, num_kv_groups,
            dropout=attn_dropout, attn_logit_cap=attn_logit_cap, causal=causal,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model, intermediate_dim, dropout=ffn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        residual = x
        h = self.attn_norm(x)
        h = self.attention(h, offset)
        x = residual + self.residual_dropout(h)

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        return residual + self.residual_dropout(h)
