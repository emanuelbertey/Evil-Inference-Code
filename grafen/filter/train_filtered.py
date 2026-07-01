"""train_filtered.py — Train a model on filtered Wikipedia data."""
import os, sys, json, pickle, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F

os.system(f"{sys.executable} -m pip install sentence-transformers -q")
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────
FILTERED_FILE = "filter/fútbol.txt"  # default, change as needed
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 3e-4
EPOCHS = 10

# ── Load filtered text ──────────────────────────────────────────────────
def load_text(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Remove comment lines
    lines = [l for l in text.split("\n") if not l.startswith("#")]
    text = "\n".join(lines)
    return text

# ── Character tokenizer (simple, no deps) ────────────────────────────────
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i+1 for i, c in enumerate(chars)}
        self.stoi["<PAD>"] = 0
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

# ── Model (tiny transformer) ─────────────────────────────────────────────
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos[:, :T, :]
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x)

# ── Train ────────────────────────────────────────────────────────────────
def train():
    text = load_text(FILTERED_FILE)
    print(f"Loaded {len(text)} chars from {FILTERED_FILE}")

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = len(data)
    n_train = int(n * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]

    model = TinyTransformer(tok.vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - SEQ_LEN - 1, (BATCH_SIZE,))
        x = torch.stack([d[i:i+SEQ_LEN] for i in ix])
        y = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix])
        return x, y

    print(f"Vocab: {tok.vocab_size} chars | Params: {sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for step in range(200):
            x, y = get_batch("train")
            logits = model(x)
            loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # Validation
        model.eval()
        with torch.no_grad():
            x, y = get_batch("val")
            logits = model(x)
            val_loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1)).item()
        model.train()
        print(f"  E{epoch+1:2d} train_loss={total_loss/200:.4f} val_loss={val_loss:.4f}")

    # Generate sample
    model.eval()
    with torch.no_grad():
        context = data[:SEQ_LEN].unsqueeze(0)
        for _ in range(100):
            logits = model(context[:, -SEQ_LEN:])
            probs = F.softmax(logits[:, -1, :] / 1.0, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            context = torch.cat([context, next_tok], dim=1)
        print(f"\nSample:\n{tok.decode(context[0].tolist())[:300]}")
    torch.save(model.state_dict(), "filter_model.pt")
    print("\nSaved: filter_model.pt")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FILTERED_FILE = sys.argv[1]
    train()
