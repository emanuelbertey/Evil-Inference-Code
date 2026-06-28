"""Load the 768/24 model from safetensors and run inference."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from python.model import TransformerLM
from safetensors.torch import load_file
from tokenizers import Tokenizer

device = torch.device("cpu")
_D = os.path.dirname(__file__)

# Load tokenizer
tok = Tokenizer.from_file(os.path.join(_D, "tokenizer.json"))
print(f"Vocab: {tok.get_vocab_size()}")

# Create model with EXACT training params
model = TransformerLM(
    vocab_size=tok.get_vocab_size(),
    d_model=768,
    num_layers=24,
    num_heads=12,
    num_kv_groups=4,
    use_swiglu=True,
    use_x0=True,
    max_seq_len=320,
    residual_dropout=0.0,
    attn_dropout=0.0,
    ffn_dropout=0.0,
).to(device).eval()

# Load weights
state = load_file(os.path.join(_D, "model_test.safetensors"))
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Missing: {missing}")
print(f"Unexpected: {unexpected}")
del state

# Encode a prompt
prompt = "desde"
ids = tok.encode(prompt).ids
print(f"Prompt ids: {ids}")
x = torch.tensor([ids], dtype=torch.long)

# Generate
with torch.no_grad():
    for i in range(50):
        logits = model.forward_train_partial_rope(x, rotary_pct=0.25)
        next_id = logits[0, -1].argmax().item()
        print(f"  {i}: {next_id} -> '{tok.decode([next_id])}'")
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)

output = tok.decode(x[0].tolist())
print(f"\nOUTPUT: {output}")
