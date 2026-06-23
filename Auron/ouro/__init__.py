"""Auron — Chimera hybrid GDN-Attention language models.

A novel architecture combining Gated Delta Networks (GDN) for efficient
recurrence with standard attention for precise recall, using a Chimera
topology: unique bottom layers for feature extraction + shared top layers
looped for iterative reasoning.

Usage:
    from ouro import load_model, generate

    model, tokenizer, device = load_model("nyxia/Auron-279M")
    generate(model, tokenizer, device, "The history of")
"""

from .config import ModelConfig, ChimeraConfig
from .model import ChimeraTransformer
from .generate import generate, load_model

__version__ = "0.1.0"
