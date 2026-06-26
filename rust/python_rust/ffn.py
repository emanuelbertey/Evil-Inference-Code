import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up
        h = self.dropout(h)
        return self.down_proj(h)
