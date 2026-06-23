"""Run inference on all 3 Auron models and save results to dossier."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/fy/Data_1/Dev/sk_ml/Ouro/.env")

sys.path.insert(0, "/home/fy/Data_1/Dev/sk_ml/Ouro")

from ouro import load_model, generate

PROMPTS = [
    "The history of artificial intelligence",
    "def fibonacci(n):",
    "In a world where machines could think,",
    "The fundamental principles of quantum mechanics",
    "Once upon a time in a small village,",
]

MODELS = [
    ("nyxia/Auron-279M", "279M"),
    ("nyxia/Auron-510M", "510M"),
    ("nyxia/Auron-1.1B", "1.1B"),
]

GEN_PARAMS = dict(
    max_tokens=250,
    temperature=0.7,
    top_k=20,
    top_p=0.95,
    rep_pen=1.0,
    presence_pen=1.5,
    stream=False,
)

DOSSIER_DIR = Path("/home/fy/Data_1/Dev/sk_ml/zara_ml/docs/dossier")
DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

for repo_id, size_label in MODELS:
    print(f"\n{'='*60}")
    print(f"Loading {repo_id}...")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(repo_id)

    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/5] Generating for prompt: {prompt!r}")
        output = generate(model, tokenizer, device, prompt, **GEN_PARAMS)
        # Strip the prompt from the output to get just completion, keep full text for display
        print(f"  -> Done ({len(output)} chars total)")
        results.append((prompt, output))

    # Write markdown file
    out_path = DOSSIER_DIR / f"{size_label}_auron_infer.md"
    lines = [
        f"# Auron-{size_label} Inference Examples",
        f"",
        f"Generated with: temp=0.7, top_k=20, top_p=0.95, rep_pen=1.0, presence_pen=1.5, max_tokens=250",
        f"",
    ]
    for prompt, output in results:
        lines.append(f'## Prompt: "{prompt}"')
        lines.append(f"")
        lines.append(output)
        lines.append(f"")
        lines.append("---")
        lines.append(f"")

    out_path.write_text("\n".join(lines))
    print(f"\nSaved -> {out_path}")

    # Unload model
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Unloaded {repo_id}")

print("\nAll done.")
