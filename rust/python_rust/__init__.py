# python_rust — Rust-compatible TransformerLM (GQA+RoPE+SwiGLU) for TPU

from .norm import RMSNorm
from .rope import apply_rope, apply_rope_partial
from .attention import repeat_kv, GQAAttention
from .ffn import SwiGLUFFN
from .block import TransformerBlock
from .model import TransformerLM
from .tokenizer import BPEWrapper

# Default config matching Rust training (transformer_quant_kv)
CONFIG = {
    "vocab_size": 16000,
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "num_kv_groups": 4,
    "max_seq_len": 128,
    "batch_size": 8,
    "stride": 128,
    "grad_accum": 2,
    "lr": 3e-4,
    "warmup_steps": 50,
    "total_steps": 100000,
    "lr_min_ratio": 0.2,
    "norm_eps": 1e-5,
    "ffn_expansion": 4.0,
    "ffn_round_to": 64,
    "attn_dropout": 0.0,
    "ffn_dropout": 0.0,
    "residual_dropout": 0.0,
    "max_norm": 1.0,
    # HuggingFace
    "hf_repo": "ScortexIA/laurelia",
    "hf_tag": "gens0x",
    "push_every_minutes": 10,
}
