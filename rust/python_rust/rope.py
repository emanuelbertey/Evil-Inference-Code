import math
import torch


def _compute_trig(q, k, offset, rotary_dim=None):
    B, S, H, D_h = q.shape
    G = k.shape[2]
    hh = (rotary_dim if rotary_dim is not None else D_h) // 2
    inv_freq = torch.tensor(
        [1.0 / (10000.0 ** (2 * i / D_h)) for i in range(hh)],
        dtype=torch.float32, device=q.device,
    )
    positions = torch.arange(offset, offset + S, dtype=torch.float32, device=q.device)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return cos, sin, hh


def _rotate_half(x, cos, sin, hh):
    x1, x2 = x[..., :hh], x[..., hh:2*hh]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, offset: int = 0):
    """Full RoPE. q: (B,S,H,D_h), k: (B,S,G,D_h)"""
    cos, sin, hh = _compute_trig(q, k, offset)
    return _rotate_half(q, cos, sin, hh), _rotate_half(k, cos, sin, hh)


def apply_rope_partial(q: torch.Tensor, k: torch.Tensor, offset: int = 0,
                       rotary_pct: float = 0.5):
    """Partial RoPE — only first rotary_pct fraction of head dims."""
    _, _, _, D_h = q.shape
    rotary_dim = int(D_h * rotary_pct) // 2 * 2
    if rotary_dim >= D_h or rotary_dim == 0:
        return apply_rope(q, k, offset) if rotary_dim >= D_h else (q, k)
    cos, sin, hh = _compute_trig(q, k, offset, rotary_dim)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_out = torch.cat([_rotate_half(q_rot, cos, sin, hh), q_pass], dim=-1)
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    k_out = torch.cat([_rotate_half(k_rot, cos, sin, hh), k_pass], dim=-1)
    return q_out, k_out
