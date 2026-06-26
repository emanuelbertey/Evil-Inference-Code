import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


def repeat_kv(x: torch.Tensor, num_heads: int, num_kv_groups: int):
    """(B,S,G,D_h) → (B,S,H,D_h) — broadcast KV for GQA."""
    if num_kv_groups == num_heads:
        return x
    repeats = num_heads // num_kv_groups
    B, S, G, D_h = x.shape
    x = x[:, :, :, None, :].expand(-1, -1, -1, repeats, -1)
    return x.reshape(B, S, num_heads, D_h)


class GQAAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_groups: int,
                 head_dim: int | None = None, dropout: float = 0.0,
                 attn_logit_cap: float | None = None, causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        self.attn_logit_cap = attn_logit_cap
        self.causal = causal

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: int = 0, cache=None):
        B, S, D = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_kv_groups, self.head_dim)
        q, k = apply_rope(q, k, offset)

        if cache is not None:
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
            S_kv = k.shape[1]
        else:
            S_kv = S

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        if self.causal and S > 1:
            mask = torch.triu(
                torch.full((S, S_kv), float("-inf"), device=x.device),
                diagonal=1 + S_kv - S,
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn_output)
