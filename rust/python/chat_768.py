"""Chat con modelo desde rust/model_test.safetensors (cache persistente)."""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from python.model import TransformerLM
from safetensors.torch import load_file
from tokenizers import Tokenizer

device = torch.device("cpu")
_RUST = os.path.dirname(os.path.dirname(__file__))
tok = Tokenizer.from_file(os.path.join(_RUST, "tokenizer.json"))
vocab = tok.get_vocab_size()

model = TransformerLM(
    vocab_size=vocab, d_model=768, num_layers=24, num_heads=12, num_kv_groups=4,
    use_swiglu=True, use_x0=True, max_seq_len=4096,
).to(device).eval()
state = load_file(os.path.join(_RUST, "model_test.safetensors"))
model.load_state_dict(state, strict=False)
print(f"Cargado safetensors con {len(state)} keys")
del state

num_layers = model.num_layers
session_offset = 0
session_caches = [None] * num_layers

print("Chat mode (salir para terminar)")
with torch.no_grad():
    while True:
        prompt = input("\n> ")
        if prompt.strip().lower() == "salir":
            break
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)
        prompt_len = x.shape[1]

        # Trim cache a 4K (keep last 3840, like Rust)
        if session_offset >= 4000:
            remove = session_offset - 3840
            for i in range(num_layers):
                if session_caches[i] is not None:
                    session_caches[i] = session_caches[i].keep_last(3840)
            session_offset -= remove

        # Prefill nuevos tokens
        logits, session_caches = model.forward_with_cache_partial(
            x, session_offset, session_caches, 0.25
        )
        session_offset += prompt_len

        generated = x.clone()
        for _ in range(50):
            next_logits = logits[:, -1, :] / 0.8
            if 40 > 0:
                vals, _ = torch.topk(next_logits, 40)
                next_logits[next_logits < vals[:, -1:]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            logits, session_caches = model.forward_with_cache_partial(
                next_token, session_offset, session_caches, 0.25
            )
            session_offset += 1

        print(tok.decode(generated[0].tolist()))
