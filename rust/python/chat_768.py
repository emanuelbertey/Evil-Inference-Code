"""Chat con modelo desde model_test.safetensors (cache persistente 4K)."""
import os, sys, time, torch
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

settings = {"max_new": 50, "temp": 1.0, "top_k": 50, "top_p": 0.95}
print("Chat mode. /help para ajustes, salir para terminar")

@torch.no_grad()
def gen(prompt):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    out = model.generate(x, max_new_tokens=settings["max_new"], temperature=settings["temp"],
                         top_k=settings["top_k"], top_p=settings["top_p"],
                         use_partial_rope=True, rotary_pct=0.25)
    dt = time.time() - t0
    n = out.shape[1] - x.shape[1]
    return tok.decode(out[0].tolist()), n, dt

while True:
    prompt = input("\n> ").strip()
    if not prompt:
        continue
    if prompt.lower() == "salir":
        break
    parts = prompt.split()
    cmd = parts[0].lower()
    if cmd in ("help", "ayuda"):
        print("ajustes: max N / len N / temp N / top_k N / top_p N")
        print(f"actual: max={settings['max_new']} temp={settings['temp']} top_k={settings['top_k']} top_p={settings['top_p']}")
        continue
    if cmd in ("max", "len") and len(parts) > 1 and parts[1].isdigit():
        settings["max_new"] = max(1, int(parts[1]))
        print(f"max_new = {settings['max_new']}")
        continue
    if cmd == "temp" and len(parts) > 1:
        settings["temp"] = max(0.01, float(parts[1]))
        print(f"temp = {settings['temp']}")
        continue
    if cmd in ("top_k", "top") and len(parts) > 1 and parts[1].isdigit():
        settings["top_k"] = max(1, int(parts[1]))
        print(f"top_k = {settings['top_k']}")
        continue
    if cmd == "top_p" and len(parts) > 1:
        settings["top_p"] = max(0.01, min(1.0, float(parts[1])))
        print(f"top_p = {settings['top_p']}")
        continue
    text, n, dt = gen(prompt)
    print(text)
    if n > 0:
        print(f"[{n} tokens en {dt:.2f}s = {n/dt:.1f} tok/s]")
