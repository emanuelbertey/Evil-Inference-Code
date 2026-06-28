"""Chat con modelo desde rust/model_test.safetensors."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from python.model import TransformerLM
from safetensors.torch import load_file
from tokenizers import Tokenizer

device = torch.device("cpu")
_RUST = os.path.dirname(os.path.dirname(__file__))  # carpeta rust/

tok = Tokenizer.from_file(os.path.join(_RUST, "tokenizer.json"))

model = TransformerLM(
    vocab_size=tok.get_vocab_size(),
    d_model=768, num_layers=24, num_heads=12, num_kv_groups=4,
    use_swiglu=True, use_x0=True, max_seq_len=320,
).to(device).eval()

state = load_file(os.path.join(_RUST, "model_test.safetensors"))
model.load_state_dict(state, strict=False)
print(f"Cargado safetensors con {len(state)} keys")
del state

print("Chat mode (salir para terminar)")
while True:
    prompt = input("\n> ")
    if prompt.strip().lower() == "salir":
        break
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9,
                         use_partial_rope=False)
    print(tok.decode(out[0].tolist()))
