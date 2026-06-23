"""
Prueba de referencia: Transformer real (HuggingFace GPT-2) con datos sintéticos.
Mismas dimensiones que test_transformer_vs_bit.rs: d_model=128, 4 layers, 4 heads, vocab=2048
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
VOCAB_SIZE = 2048
SEQ_LEN = 64
BATCH_SIZE = 4
TRAIN_STEPS = 50
LR = 3e-4

# ─── Mini Transformer en PyTorch (Pre-LN, SwiGLU, RoPE, GQA) ──────────

def precompute_rope_freqs(dim, max_len, base=10000.0):
    half = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)  # (max_len, half)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin, offset=0):
    B, S, H, D = x.shape
    half = D // 2
    cos = cos[offset:offset+S].view(1, S, 1, half).to(x.device)
    sin = sin[offset:offset+S].view(1, S, 1, half).to(x.device)
    x_first = x[..., :half]
    x_second = x[..., half:]
    out_first = x_first * cos - x_second * sin
    out_second = x_first * sin + x_second * cos
    return torch.cat([out_first, out_second], dim=-1)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        denom = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x / denom * self.weight

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, ffn_dim, bias=False)
        self.up = torch.nn.Linear(d_model, ffn_dim, bias=False)
        self.down = torch.nn.Linear(ffn_dim, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups=2, max_seq_len=256):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_model // num_heads
        self.num_kv_heads = num_kv_groups

        self.q_proj = torch.nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, offset=0):
        B, S, D = x.shape
        H, G, Dh = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, Dh)
        k = self.k_proj(x).view(B, S, G, Dh)
        v = self.v_proj(x).view(B, S, G, Dh)

        q = apply_rope(q, self.cos, self.sin, offset)
        k = apply_rope(k, self.cos, self.sin, offset)

        # GQA: repeat K,V heads
        k = k.repeat_interleave(H // G, dim=2)
        v = v.repeat_interleave(H // G, dim=2)

        scale = Dh ** -0.5
        attn = torch.einsum("bshd,bthd->bhst", q, k) * scale

        # Causal mask
        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn, dim=-1)
        out = torch.einsum("bhst,bthd->bshd", attn_weights, v).reshape(B, S, H * Dh)
        return self.o_proj(out)

class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, ffn_dim, max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, num_heads, num_kv_groups, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)
    def forward(self, x, offset=0):
        x = x + self.attn(self.attn_norm(x), offset)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class TransformerLM(torch.nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_kv_groups, vocab_size, max_seq_len=256):
        super().__init__()
        ffn_dim = int(4 * d_model * 2 / 3 / 64 + 1) * 64
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerLayer(d_model, num_heads, num_kv_groups, ffn_dim, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, 0)
        x = self.final_norm(x)
        return self.head(x)

# ─── Dataset sintético ────────────────────────────────────────────────

class RandomCopyDataset(Dataset):
    """Copy task: input [1,2,3,4,0,0,0,0] -> target [0,0,0,0,1,2,3,4]"""
    def __init__(self, num_samples, vocab_size, seq_len):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # Random data (like our Rust test uses)
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = x.clone()
        return x, y

# ─── Test 1: Datos aleatorios (como Rust) ─────────────────────────────

def test_random_data():
    print("="*60)
    print("Test 1: Datos aleatorios (como test_transformer_vs_bit.rs)")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(D_MODEL, NUM_LAYERS, NUM_HEADS, 2, VOCAB_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    dataset = RandomCopyDataset(100, VOCAB_SIZE, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    step = 0
    for epoch in range(10):
        for x, y in loader:
            if step >= TRAIN_STEPS:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 10 == 0:
                print(f"  step {step+1:3d}/{TRAIN_STEPS}  loss {loss.item():.6f}")
            step += 1

    print(f"\n  Resultado: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  {'✓ Aprende' if losses[-1] < losses[0] * 0.9 else '✗ No aprende'}")
    print()
    return losses

# ─── Test 2: Copy task (señal real) ───────────────────────────────────

def test_copy_task():
    print("="*60)
    print("Test 2: Copy task (señal real - debe llegar a loss ~0)")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(D_MODEL, NUM_LAYERS, NUM_HEADS, 2, VOCAB_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Copy task: input [a,b,c,d] -> target [a,b,c,d] (shifted by model internally)
    # We'll just do next-token prediction on deterministic patterns
    patterns = torch.randint(0, VOCAB_SIZE, (100, SEQ_LEN // 2))
    x = torch.cat([patterns, torch.zeros(100, SEQ_LEN // 2, dtype=torch.long)], dim=1)
    y = torch.cat([torch.zeros(100, SEQ_LEN // 2, dtype=torch.long), patterns], dim=1)

    dataset = torch.utils.data.TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    step = 0
    for epoch in range(50):
        for xb, yb in loader:
            if step >= TRAIN_STEPS:
                break
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 10 == 0:
                print(f"  step {step+1:3d}/{TRAIN_STEPS}  loss {loss.item():.6f}")
            step += 1

    print(f"\n  Resultado: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  {'✓ Aprende (copy task)' if losses[-1] < 1.0 else '✗ No aprende'}")
    print()
    return losses

# ─── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    l1 = test_random_data()
    l2 = test_copy_task()

    print("="*60)
    print("RESUMEN")
    print("="*60)
    print(f"Test 1 (random data): {l1[0]:.4f} -> {l1[-1]:.4f}")
    print(f"Test 2 (copy task):   {l2[0]:.4f} -> {l2[-1]:.4f}")
