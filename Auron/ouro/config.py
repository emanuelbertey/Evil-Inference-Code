"""Model configuration for Chimera GDN/Attention hybrid architecture.

Config is loaded from the HuggingFace repo's config.json — no presets here.
"""

import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Base configuration for GDN/Attention hybrid transformer."""

    dim: int = 512
    n_layers: int = 8
    vocab_size: int = 151936
    max_seq_len: int = 2048
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    gdn_expand_v: int = 1
    gdn_head_dim: int = 64
    gdn_n_heads: int = 8
    conv_kernel: int = 4
    gdn_use_gate: bool = True
    gdn_use_short_conv: bool = True
    ffn_mult: float = 2.5
    attn_interval: int = 4
    use_x0_inject: bool = True
    use_resid_lambdas: bool = True
    use_skip_connections: bool = True
    use_diff_attn: bool = False
    rope_base: float = 10000.0
    partial_rotary_factor: float = 0.25

    @property
    def ffn_hidden(self) -> int:
        return int(self.dim * self.ffn_mult)

    @property
    def n_gdn_layers(self) -> int:
        return self.n_layers - self.n_attn_layers

    @property
    def n_attn_layers(self) -> int:
        return self.n_layers // self.attn_interval

    def is_attn_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.attn_interval == 0


@dataclass
class ChimeraConfig(ModelConfig):
    """Chimera Stack: unique bottom layers + shared top layers looped."""
    n_bottom: int = 4
    n_physical_top: int = 4
    n_top_loops: int = 3

    def __post_init__(self):
        self.n_layers = self.n_bottom + self.n_physical_top * self.n_top_loops


def from_dict(d: dict) -> ChimeraConfig:
    """Construct ChimeraConfig from a HF config.json dict."""
    import dataclasses
    valid = {f.name for f in dataclasses.fields(ChimeraConfig)}
    valid |= {f.name for f in dataclasses.fields(ModelConfig)}
    return ChimeraConfig(**{k: v for k, v in d.items() if k in valid})
