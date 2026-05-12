import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rope(x: torch.Tensor, freqs: torch.Tensor, offset: int = 0):
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs[offset:offset + S].view(1, 1, S, D // 2)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return x_rotated.type_as(x)

def unpack_tq2_0(qs: torch.Tensor) -> torch.Tensor:
    chunks = qs.view(*qs.shape[:-1], qs.shape[-1] // 32, 32)
    vals = torch.stack(
        [
            (chunks & 0x03).to(torch.float32) - 1.0,
            ((chunks >> 2) & 0x03).to(torch.float32) - 1.0,
            ((chunks >> 4) & 0x03).to(torch.float32) - 1.0,
            ((chunks >> 6) & 0x03).to(torch.float32) - 1.0,
        ],
        dim=-2,
    )
    return vals.reshape(*qs.shape[:-1], qs.shape[-1] * 4)

def unpack_q1_0(qs: torch.Tensor) -> torch.Tensor:
    bits = [
        ((qs >> shift) & 0x01).to(torch.float32) * 2.0 - 1.0
        for shift in range(8)
    ]
    return torch.stack(bits, dim=-1).reshape(*qs.shape[:-1], qs.shape[-1] * 8)

class Q1_0_Embedding(nn.Module):
    QK = 32
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_blocks = (vocab_size * dim) // self.QK
        self.register_buffer('blocks_d', torch.zeros(self.num_blocks))
        self.register_buffer('blocks_qs', torch.zeros((self.num_blocks, 4), dtype=torch.uint8))

    def forward(self, x):
        blocks_per_row = self.dim // self.QK
        d = self.blocks_d.view(self.vocab_size, blocks_per_row)[x]
        qs = self.blocks_qs.view(self.vocab_size, blocks_per_row, 4)[x]
        w = unpack_q1_0(qs)
        return (w * d.unsqueeze(-1)).view(*x.shape, self.dim)

    def dequant_rows(self, start: int, end: int) -> torch.Tensor:
        blocks_per_row = self.dim // self.QK
        d = self.blocks_d.view(self.vocab_size, blocks_per_row, 1)[start:end]
        qs = self.blocks_qs.view(self.vocab_size, blocks_per_row, 4)[start:end]
        return (unpack_q1_0(qs) * d).view(end - start, self.dim)

    def linear(self, x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        x_flat = x.reshape(-1, self.dim)
        pieces = []
        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            pieces.append(F.linear(x_flat, self.dequant_rows(start, end)))
        return torch.cat(pieces, dim=-1).view(*x.shape[:-1], self.vocab_size)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class TernaryEmbedding(nn.Module):
    QK = 256
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_blocks = (vocab_size * dim) // self.QK
        self.register_buffer('blocks_d', torch.zeros(self.num_blocks))
        self.register_buffer('blocks_qs', torch.zeros((self.num_blocks, 64), dtype=torch.uint8))

    def forward(self, x):
        blocks_per_row = self.dim // self.QK
        d = self.blocks_d.view(self.vocab_size, blocks_per_row)[x]
        qs = self.blocks_qs.view(self.vocab_size, blocks_per_row, 64)[x]
        w = unpack_tq2_0(qs)
        return (w * d.unsqueeze(-1)).view(*x.shape, self.dim)

    def dequant_rows(self, start: int, end: int) -> torch.Tensor:
        blocks_per_row = self.dim // self.QK
        d = self.blocks_d.view(self.vocab_size, blocks_per_row, 1)[start:end]
        qs = self.blocks_qs.view(self.vocab_size, blocks_per_row, 64)[start:end]
        return (unpack_tq2_0(qs) * d).view(end - start, self.dim)

    def linear(self, x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        x_flat = x.reshape(-1, self.dim)
        pieces = []
        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            pieces.append(F.linear(x_flat, self.dequant_rows(start, end)))
        return torch.cat(pieces, dim=-1).view(*x.shape[:-1], self.vocab_size)

class TQ2_0_Linear(nn.Module):
    QK = 256
    def __init__(self, in_features: int, out_features: int, bias: bool = False, lazy=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks_per_row = in_features // self.QK
        self.num_blocks = out_features * self.num_blocks_per_row
        self.register_buffer('blocks_d', torch.zeros(self.num_blocks))
        self.register_buffer('blocks_qs', torch.zeros((self.num_blocks, 64), dtype=torch.uint8))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        d = self.blocks_d.view(self.out_features, self.num_blocks_per_row, 1)
        qs = self.blocks_qs.view(self.out_features, self.num_blocks_per_row, 64)
        w_bits = unpack_tq2_0(qs)
        w = (w_bits * d).view(self.out_features, self.in_features)
        return F.linear(x, w, self.bias)

class Q1_0_Linear(nn.Module):
    QK = 32
    def __init__(self, in_features: int, out_features: int, bias: bool = False, lazy=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks_per_row = in_features // self.QK
        self.num_blocks = out_features * self.num_blocks_per_row
        self.register_buffer('blocks_d', torch.zeros(self.num_blocks))
        self.register_buffer('blocks_qs', torch.zeros((self.num_blocks, 4), dtype=torch.uint8))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        d = self.blocks_d.view(self.out_features, self.num_blocks_per_row, 1)
        qs = self.blocks_qs.view(self.out_features, self.num_blocks_per_row, 4)
        w_bits = unpack_q1_0(qs)
        w = (w_bits * d).view(self.out_features, self.in_features)
        return F.linear(x, w, self.bias)

def make_linear(in_features, out_features, bias=False, quant_mode="tq2_0", lazy=True):
    if quant_mode == "q1_0":
        return Q1_0_Linear(in_features, out_features, bias, lazy=lazy)
    return TQ2_0_Linear(in_features, out_features, bias, lazy=lazy)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, quant_mode: str = "tq2_0", norm_eps: float = 1e-5, lazy=True):
        super().__init__()
        self.w1 = make_linear(dim, hidden_dim, quant_mode=quant_mode, lazy=lazy)
        self.w2 = make_linear(hidden_dim, dim, quant_mode=quant_mode, lazy=lazy)
        self.w3 = make_linear(dim, hidden_dim, quant_mode=quant_mode, lazy=lazy)
        self.ffn_sub_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.use_ffn_sub_norm = False

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w3(x)
        if self.use_ffn_sub_norm:
            h = self.ffn_sub_norm(h)
        return self.w2(h)
