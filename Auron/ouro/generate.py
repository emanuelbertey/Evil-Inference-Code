"""Load and generate from Auron models hosted on HuggingFace."""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ChimeraConfig, from_dict
from .model import ChimeraTransformer


def load_model(
    repo_id: str,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
    token: str = None,
) -> tuple:
    """Load an Auron model from a HuggingFace repo.

    Args:
        repo_id: HF repo (e.g. "nyxia/Auron-279M")
        device: "auto", "cpu", "cuda", or "cuda:0"
        dtype: torch.bfloat16 (default) or torch.float32
        token: HF token for private repos (or set HF_TOKEN env var)

    Returns:
        (model, tokenizer, device)
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    token = token or os.environ.get("HF_TOKEN")

    # Download config
    config_path = hf_hub_download(repo_id, "config.json", token=token)
    with open(config_path) as f:
        config_data = json.load(f)
    cfg = from_dict(config_data)

    # Build model
    model = ChimeraTransformer(cfg)

    # Load weights
    weights_path = hf_hub_download(repo_id, "model.safetensors", token=token)
    from safetensors.torch import load_file
    state_dict = load_file(weights_path)

    # Handle tied weights (embed.weight == lm_head.weight stored separately)
    if "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
    model.load_state_dict(state_dict, strict=False)

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device=device, dtype=dtype).eval()

    # Load tokenizer from same repo
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token, trust_remote_code=True)

    step = config_data.get("step", "?")
    size = config_data.get("size_label", "?")
    print(f"Loaded {repo_id} ({size}, step {step}) on {device}")

    return model, tokenizer, device


@torch.no_grad()
def generate(
    model: ChimeraTransformer,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.95,
    rep_pen: float = 1.0,
    presence_pen: float = 1.5,
    stream: bool = True,
) -> str:
    """Generate text with Qwen-style sampling.

    Default params (Qwen 3 casual):
        temp=0.7, top_k=20, top_p=0.95, rep_pen=1.0, presence_pen=1.5

    Args:
        model: ChimeraTransformer instance
        tokenizer: HF tokenizer
        device: torch device
        prompt: input text
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k: top-k filtering (0 = disabled)
        top_p: nucleus sampling threshold
        rep_pen: repetition penalty on context tokens (multiplicative)
        presence_pen: presence penalty on generated tokens (additive)
        stream: print tokens as generated

    Returns:
        Generated text (prompt + completion)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = set()
    output_tokens = []

    if stream:
        print(prompt, end="", flush=True)

    for _ in range(max_tokens):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)

        next_logits = logits[0, -1, :].clone()

        # Repetition penalty (multiplicative, on context tokens)
        if rep_pen > 1.0:
            for tid in set(input_ids[0].tolist()):
                if next_logits[tid] < 0:
                    next_logits[tid] *= rep_pen
                else:
                    next_logits[tid] /= rep_pen

        # Presence penalty (additive, on all previously generated tokens)
        if presence_pen != 0:
            for tid in generated_ids:
                next_logits[tid] -= presence_pen

        # Temperature
        next_logits = next_logits / temperature

        # Top-k
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = -float("Inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cum_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = 0
            indices_to_remove = sorted_remove.scatter(
                dim=0, index=sorted_indices, src=sorted_remove
            )
            next_logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        generated_ids.add(next_token.item())
        output_tokens.append(next_token.item())

        token_str = tokenizer.decode(next_token.item())
        if stream:
            print(token_str, end="", flush=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

    if stream:
        print()

    return tokenizer.decode(input_ids[0].tolist())
