# Auron

Chimera hybrid GDN-Attention language models — a novel architecture combining [Gated Delta Networks](https://arxiv.org/abs/2412.06464) for O(n) recurrent processing with standard GQA attention for precise recall.

**Paper:** [Auron: Depth-Efficient Language Models via Hybrid Recurrent-Attention Weight Sharing](https://arxiv.org/abs/TODO)
**Models:** [huggingface.co/nyxia](https://huggingface.co/nyxia)
**Author:** [Florian Gasquez](https://fyx.jp) ([Soulkyn](https://soulkyn.com))

## Architecture

The **Chimera topology** splits the network into two functional zones:

- **Bottom ("Retina"):** Unique layers for dense token parsing — feature extraction before reasoning
- **Top ("Brain"):** Shared physical blocks looped N times — weight sharing as learned recurrence

Both sections interleave GDN and Attention layers at a 3:1 ratio (`attn_interval=4`), with SwiGLU FFN on every layer. Convergence accelerators include x0 residual injection, learnable residual lambdas, U-Net skip connections, and per-loop layer ID gate injection.

The shared top blocks fit in GPU L2 cache (35MB for 510M on H100's 50MB L2), making Ouroboros loops execute with near-zero memory latency.

## Models

| Model | Total Params | Virtual Params | Topology | Context |
|-------|-------------|----------------|----------|---------|
| [Auron-279M](https://huggingface.co/nyxia/Auron-279M) | 279M | ~350M | 4 bottom + 4×3 top (dim=1024) | 2048 |
| [Auron-510M](https://huggingface.co/nyxia/Auron-510M) | 510M | ~787M | 4 bottom + 4×3 top (dim=1536) | 2048 |
| [Auron-1.1B](https://huggingface.co/nyxia/Auron-1.1B) | 1.1B | ~1.8B | 6 bottom + 6×3 top (dim=2048) | 2048 |

All models use the Qwen 3 tokenizer (151,936 vocab), partial RoPE (25%), GDN V expansion 2×, and tied input/output embeddings.

## Install

```bash
# With rye
rye sync

# Or pip
pip install -e .
```

## Usage

### CLI

```bash
# Set your HF token (for private repos)
echo "HF_TOKEN=hf_your_token" > .env

# Generate
ouro "The history of"
ouro "def fibonacci(n):" --model nyxia/Auron-510M
ouro "Scientists have" --temp 1.0 --max-tokens 200
```

### Python

```python
from ouro import load_model, generate

model, tokenizer, device = load_model("nyxia/Auron-279M")
generate(model, tokenizer, device, "The history of")
```

### Sampling Parameters

Default Qwen Casual params — optimal for Auron at current training stage:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temp` | 0.7 | Sampling temperature (min ~0.6 to avoid attractor wells) |
| `top_k` | 20 | Top-k filtering |
| `top_p` | 0.9 | Nucleus sampling |
| `rep_pen` | 1.0 | Repetition penalty (multiplicative, on context) |
| `presence_pen` | 1.5 | Presence penalty (additive, on generated tokens — required for Ouroboros) |

> **Note:** The Ouroboros weight-sharing creates "attractor wells" — stronger token persistence than standard transformers. Presence penalty >= 1.5 is required to prevent repetition loops. This is an architectural property, not a training artifact.

## Training

Pretrained on a mixed dataset (~5B tokens total):
- **75%** FineWeb-Edu — general knowledge
- **18%** StarCoderData — code (Python, JSON, Markdown)
- **5%** FineMath-4+ — mathematics
- **2%** UltraChat 200k — conversational structure (ChatML)

Optimizer: Muon (2D weights) + AdamW (embeddings at 4e-4, scalars at 8e-5). Schedule: WSD with 10% cosine warmdown.

Tokenizer: Qwen 3 (151,936 tokens) with patched chat template allowing system messages at any position.

## License

CC BY 4.0 (paper) / Apache 2.0 (code)
