"""Chat con modelo desde rust/model_test.safetensors (cache persistente)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from python.model import TransformerLM
from safetensors.torch import load_file
from tokenizers import Tokenizer

device = torch.device("cpu")
_RUST = os.path.dirname(os.path.dirname(__file__))
tok_path = os.path.join(_RUST, "tokenizer.json")
if not os.path.exists(tok_path):
    tok_path = os.path.join(_RUST, "tokenizer.json")
tok = Tokenizer.from_file(tok_path)
vocab = tok.get_vocab_size()

model = TransformerLM(
    vocab_size=vocab, d_model=768, num_layers=24, num_heads=12, num_kv_groups=4,
    use_swiglu=True, use_x0=True, max_seq_len=4096,
).to(device).eval()
state = load_file(os.path.join(_RUST, "model_test.safetensors"))
model.load_state_dict(state, strict=False)
print(f"Cargado model_test.safetensors ({len(state)} tensors)")

print("Chat mode (salir para terminar)")
with torch.no_grad():
    while True:
        prompt = input("\n> ")
        if prompt.strip().lower() == "salir":
            break
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.95,
                             use_partial_rope=True, rotary_pct=0.25)
        print(tok.decode(out[0].tolist()))
