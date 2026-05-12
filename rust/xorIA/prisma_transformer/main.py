import torch
from config import PrismaConfig
from transformer import PrismaTransformer

def simulate_generation(model, initial_tokens, max_new_tokens=3, mode_name=""):
    print(f"--- BitNet Transformer [{mode_name}] ---")

    print(f"[Prompt] Evaluando prompt de longitud {initial_tokens.shape[1]}")
    logits, kv_caches = model(initial_tokens, kv_caches=None, offset=0)

    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
    print(f"Token predicto: {next_token.item()} | KV Cache K={kv_caches[0][0].shape}")

    current_offset = initial_tokens.shape[1]

    for i in range(max_new_tokens):
        print(f"\n[Generacion] Paso {i+1}: Evaluando solo 1 token nuevo...")
        logits, kv_caches = model(next_token, kv_caches=kv_caches, offset=current_offset)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        current_offset += 1
        print(f"Token predicto: {next_token.item()} | KV Cache K={kv_caches[0][0].shape}")

if __name__ == "__main__":
    prompt_tokens = torch.randint(0, 1000, (1, 5))

    # ========== MODO 1: Q1_0 — Prisma 1 bit puro (+1/-1) ==========
    print("=" * 60)
    config_q1 = PrismaConfig(dim=32, n_layers=2, n_heads=4, vocab_size=1000, quant_mode="q1_0")
    model_q1 = PrismaTransformer(config_q1)
    simulate_generation(model_q1, prompt_tokens, max_new_tokens=2, mode_name="Q1_0 — 1 bit (+1/-1)")

    print()

    # ========== MODO 2: TQ2_0 — Ternario 1.58 bit (-1/0/+1) ==========
    print("=" * 60)
    config_tq = PrismaConfig(dim=32, n_layers=2, n_heads=4, vocab_size=1000, quant_mode="tq2_0")
    model_tq = PrismaTransformer(config_tq)
    simulate_generation(model_tq, prompt_tokens, max_new_tokens=2, mode_name="TQ2_0 — 1.58 bit (-1/0/+1)")
