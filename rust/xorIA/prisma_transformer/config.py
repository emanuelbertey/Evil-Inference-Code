from dataclasses import dataclass

@dataclass
class PrismaConfig:
    dim: int = 2048
    n_layers: int = 28
    n_heads: int = 16
    n_heads_kv: int = 8      # Soporte para GQA (Grouped Query Attention)
    vocab_size: int = 151936 # Tamaño aproximado de Qwen2/3
    hidden_dim: int = 6144   # FFN hidden dim
    norm_eps: float = 1e-6
    max_seq_len: int = 32768
    rope_theta: float = 1000000.0 # Base masiva de Qwen
    bias: bool = False
    quant_mode: str = "tq2_0"
