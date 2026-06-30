"""Verify that python/train.py init does NOT have logit over-amplification bug.

Detects:
  1. Logit std > 0.8 → softmax near one-hot → model memorizes
  2. Logit std < 0.01 → too uniform, gradients vanish

python/model.py uses PyTorch defaults (no custom init):
  - Embedding: N(0, 1)
  - Linear: Kaiming Uniform with a=sqrt(5)
  - No weight tying

Pass if: 0.01 <= logit_std < 0.8 and logit_max < 5.0
"""

import sys, os, math, torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, ".."))
from python.model import TransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_model(d_model, num_layers, num_heads, vocab_size, tag="", num_kv_groups=None):
    if num_kv_groups is None:
        num_kv_groups = 2
        while num_heads % num_kv_groups != 0 and num_kv_groups > 1:
            num_kv_groups -= 1
    m = TransformerLM(
        vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, num_kv_groups=num_kv_groups,
        use_swiglu=True, use_x0=False, max_seq_len=128,
        residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    ).to(device)
    x = torch.randint(0, vocab_size, (4, 64), device=device)
    with torch.no_grad():
        logits = m(x)
        std, mx = logits.std().item(), logits.max().item()
        ok_std = 0.01 <= std < 0.8
        ok_mx = mx < 5.0
        ok = ok_std and ok_mx
        issues = []
        if std >= 0.8: issues.append("OVERFIT BUG")
        if std < 0.01: issues.append("TOO UNIFORM")
        if mx >= 5.0: issues.append("EXTREME MAX")
        tag_issue = f"  ({', '.join(issues)})" if issues else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag or f'd_model={d_model} vocab={vocab_size}'}: "
              f"std={std:.4f}  max={mx:.3f}{tag_issue}")
        return ok

def main():
    tests = 0
    passed = 0
    print("=== Init verification: python/train.py (no MLA, PyTorch defaults) ===\n")

    # Config matching python/train.py
    tests += 1
    if check_model(d_model=768, num_layers=24, num_heads=12, vocab_size=16000,
                   num_kv_groups=4,
                   tag="train.py (d_model=768, 24lay, vocab=16k)"):
        passed += 1

    # Small model
    tests += 1
    if check_model(d_model=256, num_layers=3, num_heads=4, vocab_size=65,
                   tag="small (d_model=256, 3lay, vocab=65)"):
        passed += 1

    tests += 1
    if check_model(d_model=256, num_layers=3, num_heads=4, vocab_size=50000,
                   tag="small+bigvocab (d_model=256, vocab=50k)"):
        passed += 1

    # Edge: large model
    tests += 1
    if check_model(d_model=1024, num_layers=12, num_heads=16, vocab_size=32000,
                   num_kv_groups=4,
                   tag="edge (d_model=1024, 12lay, vocab=32k)"):
        passed += 1

    print(f"\n{'='*50}")
    print(f"Result: {passed}/{tests} passed")
    if passed == tests:
        print("OK — No init bug detected.")
    else:
        print("INIT PROBLEM DETECTED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
