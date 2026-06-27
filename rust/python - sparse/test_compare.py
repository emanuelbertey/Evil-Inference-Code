"""Test: run forward on fixed text, save logits, compare with Rust later."""
import os, sys, json, struct
import torch
import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from python.model import TransformerLM

device = torch.device("cpu")
text = "The king is in the castle"
seq_len = len(text.split())  # simple whitespace split for test

# Load config and model
model = TransformerLM(d_model=256, num_layers=6, num_heads=8, num_kv_groups=4,
                       vocab_size=16000, max_seq_len=128, use_swiglu=True).to(device)

safetensors_path = os.path.join(os.path.dirname(__file__), "model_test.safetensors")
from safetensors.torch import load_file
state = load_file(safetensors_path)
model.load_state_dict(state, strict=False)
del state

model.eval()

# Tokenize
tokenizer = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "tokenizer.json"))
ids = tokenizer.encode(text).ids
print(f"Text: {text}")
print(f"Input IDs ({len(ids)}): {ids}")

# Forward
input_ids = torch.tensor([ids], dtype=torch.long)
with torch.no_grad():
    logits = model(input_ids)  # (1, seq_len, vocab_size)

print(f"Logits shape: {logits.shape}")

# Save: logits as raw f32, input_ids as int32
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
os.makedirs(out_dir, exist_ok=True)

logits_np = logits.cpu().numpy().astype(np.float32)
np.save(os.path.join(out_dir, "logits.npy"), logits_np)
np.save(os.path.join(out_dir, "input_ids.npy"), np.array(ids, dtype=np.int32))
np.save(os.path.join(out_dir, "text.npy"), np.array([text]))

# Also save first 10 logits of last token for quick glance
last = logits_np[0, -1, :10]
print(f"First 10 logits (last token): {last}")
print(f"\nSaved to {out_dir}/")
