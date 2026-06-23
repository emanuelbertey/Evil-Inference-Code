"""Chimera GDN/Attention hybrid transformer — inference only.

Architecture:
- GDN layers (Gated Delta Net) for O(n) recurrent processing
- GQA attention layers for precise recall
- 3:1 GDN:Attn ratio (configurable via attn_interval)
- SwiGLU FFN on every layer
- Convergence accelerators: x0 injection, residual lambdas, U-Net skips
- Chimera topology: unique bottom + shared top looped
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fla.layers import GatedDeltaNet

from .config import ModelConfig, ChimeraConfig


# ── Building blocks ─────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), eps=self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype):
        return (
            self.cos_cached[:seq_len].to(dtype),
            self.sin_cached[:seq_len].to(dtype),
        )


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half_d = cos.shape[-1]
    rope_dim = half_d * 2
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([out_rot, x_pass], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """GQA attention with partial RoPE."""
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.rope_dim = int(cfg.head_dim * cfg.partial_rotary_factor) * 2
        self.rotary = RotaryEmbedding(
            self.rope_dim, max_seq_len=cfg.max_seq_len, base=cfg.rope_base
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(T, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class DiffAttention(nn.Module):
    """Differential Attention — pairs heads and subtracts for noise cancellation."""
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_diff_heads = cfg.n_heads // 2
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_diff_heads * cfg.head_dim, cfg.dim, bias=False)

        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * max(layer_idx, 1))
        self.lambda_q1 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)
        self.lambda_init = lambda_init

        self.rope_dim = int(cfg.head_dim * cfg.partial_rotary_factor) * 2
        self.rotary = RotaryEmbedding(
            self.rope_dim, max_seq_len=cfg.max_seq_len, base=cfg.rope_base
        )
        self.sub_norm = RMSNorm(cfg.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_diff_heads * 2, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(T, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        q1, q2 = q.chunk(2, dim=1)
        attn1 = F.scaled_dot_product_attention(q1, k, v, is_causal=True, enable_gqa=True)
        attn2 = F.scaled_dot_product_attention(q2, k, v, is_causal=True, enable_gqa=True)

        lam = (self.lambda_q1 * self.lambda_k1).sum() - (self.lambda_q2 * self.lambda_k2).sum() + self.lambda_init
        y = self.sub_norm(attn1 - lam * attn2)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


# ── Transformer block ──────────────────────────────────────────


class HybridBlock(nn.Module):
    """A single transformer block — GDN or Attention based on config."""

    def __init__(self, cfg: ModelConfig, layer_idx: int, force_attn: Optional[bool] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attn = force_attn if force_attn is not None else cfg.is_attn_layer(layer_idx)

        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)

        if self.is_attn:
            if cfg.use_diff_attn:
                self.mixer = DiffAttention(cfg, layer_idx)
            else:
                self.mixer = Attention(cfg, layer_idx)
        else:
            self.mixer = GatedDeltaNet(
                hidden_size=cfg.dim,
                num_heads=cfg.gdn_n_heads,
                head_dim=cfg.gdn_head_dim,
                expand_v=cfg.gdn_expand_v,
                use_gate=cfg.gdn_use_gate,
                use_short_conv=cfg.gdn_use_short_conv,
                conv_size=cfg.conv_kernel,
                mode="chunk",
                layer_idx=layer_idx,
            )

        self.ffn = SwiGLU(cfg.dim, cfg.ffn_hidden)

        if cfg.use_resid_lambdas:
            self.resid_attn = nn.Parameter(torch.full((), 1.1 ** 0.5))
            self.resid_mlp = nn.Parameter(torch.full((), 1.1 ** 0.5))
        else:
            self.resid_attn = 1.0
            self.resid_mlp = 1.0

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None,
                layer_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        if layer_id is not None and not self.is_attn:
            h = h + layer_id
        h = self.mixer(h) if self.is_attn else self.mixer(h)[0]
        x = self.resid_attn * x + h
        x = self.resid_mlp * x + self.ffn(self.norm2(x))
        return x


# ── Chimera Transformer ───────────────────────────────────────


class ChimeraTransformer(nn.Module):
    """Chimera Stack: unique bottom + ouroboros top.

    Bottom: fully unique HybridBlocks for feature extraction.
    Top: shared physical blocks looped N times for iterative reasoning.
    """

    def __init__(self, cfg: ChimeraConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.embed_norm = RMSNorm(cfg.dim)

        # Bottom: unique blocks (global virtual indices)
        self.bottom_blocks = nn.ModuleList([
            HybridBlock(cfg, i) for i in range(cfg.n_bottom)
        ])

        # Top: shared physical blocks (local indices, pattern repeats per loop)
        self.top_blocks = nn.ModuleList([
            HybridBlock(cfg, i) for i in range(cfg.n_physical_top)
        ])

        # Per-loop layer IDs for top section
        self.top_layer_ids = nn.Parameter(
            torch.zeros(cfg.n_top_loops, cfg.n_physical_top, cfg.dim)
        )

        # x0 injection (per virtual layer)
        if cfg.use_x0_inject:
            self.x0_lambdas = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(cfg.n_layers)
            ])
        else:
            self.x0_lambdas = None

        # U-Net skip connections
        if cfg.use_skip_connections and cfg.n_layers >= 4:
            n_skips = cfg.n_layers // 2
            self.skip_weights = nn.ParameterList([
                nn.Parameter(torch.ones(cfg.dim)) for _ in range(n_skips)
            ])
        else:
            self.skip_weights = None

        # LM head (tied with embedding)
        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns logits (B, T, vocab_size)."""
        B, T = input_ids.shape
        x = self.embed(input_ids)
        x = self.embed_norm(x)
        x0 = x

        skips = []
        half = self.cfg.n_layers // 2
        v = 0

        # Bottom: unique blocks
        for block in self.bottom_blocks:
            if self.x0_lambdas is not None:
                x = x + self.x0_lambdas[v] * x0
            if self.skip_weights is not None and v < half:
                skips.append(x)
            x = block(x, x0)
            if self.skip_weights is not None and v >= half:
                skip_idx = self.cfg.n_layers - 1 - v
                if skip_idx < len(skips):
                    x = x + self.skip_weights[skip_idx] * skips[skip_idx]
            v += 1

        # Top: ouroboros loop
        for loop in range(self.cfg.n_top_loops):
            for b, block in enumerate(self.top_blocks):
                if self.x0_lambdas is not None:
                    x = x + self.x0_lambdas[v] * x0
                if self.skip_weights is not None and v < half:
                    skips.append(x)
                layer_id = self.top_layer_ids[loop, b]
                x = block(x, x0, layer_id=layer_id.to(x.dtype))
                if self.skip_weights is not None and v >= half:
                    skip_idx = self.cfg.n_layers - 1 - v
                    if skip_idx < len(skips):
                        x = x + self.skip_weights[skip_idx] * skips[skip_idx]
                v += 1

        x = self.final_norm(x)
        return self.lm_head(x)
