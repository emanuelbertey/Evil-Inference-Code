"""transformer.py — Train a simple network on KG entity relationships."""
import os, sys, pickle, json, math, random
from collections import defaultdict
os.system(f"{sys.executable} -m pip install torch scikit-learn -q")
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

CACHE_FILE = "knowledge_graph.pkl"

if not os.path.exists(CACHE_FILE):
    print(f"Run procesar.py first — {CACHE_FILE} not found")
    sys.exit(1)

with open(CACHE_FILE, "rb") as f:
    kg = pickle.load(f)

entity_names = kg["entity_names"]
cooc = kg["cooc"]
hierarchy = kg["hierarchy"]
cluster_names = kg["cluster_names"]

# ── Prepare data ─────────────────────────────────────────────────────────
print(f"Preparing data: {len(entity_names)} entities, {len(cooc)} relationships")

# Entity → index
e2i = {e: i for i, e in enumerate(entity_names)}
n_entities = len(entity_names)

# Adjacency matrix (co-occurrence)
adj = torch.zeros((n_entities, n_entities), dtype=torch.float32)
for (a, b), w in cooc.items():
    if a in e2i and b in e2i:
        adj[e2i[a], e2i[b]] = w
        adj[e2i[b], e2i[a]] = w

# Node features: identity for now (learned embeddings)
node_feats = torch.eye(n_entities)

# Train: predict missing edges (link prediction)
edges = [(e2i[a], e2i[b], w) for (a, b), w in cooc.items() if a in e2i and b in e2i]
random.shuffle(edges)

# Generate negative samples (non-edges)
neg_edges = []
n_pos = len(edges)
while len(neg_edges) < n_pos:
    i, j = random.randint(0, n_entities - 1), random.randint(0, n_entities - 1)
    if i != j and adj[i, j] == 0:
        neg_edges.append((i, j, 0))

all_edges = [(i, j, 1) for i, j, w in edges] + neg_edges
random.shuffle(all_edges)

edges_t = torch.tensor([(i, j) for i, j, _ in all_edges], dtype=torch.long)
labels_t = torch.tensor([l for _, _, l in all_edges], dtype=torch.float32)

n_train = int(len(all_edges) * 0.8)
train_e, test_e = edges_t[:n_train], edges_t[n_train:]
train_l, test_l = labels_t[:n_train], labels_t[n_train:]

# ── Model ────────────────────────────────────────────────────────────────
class EntityEncoder(nn.Module):
    def __init__(self, n_entities, dim=64):
        super().__init__()
        self.embed = nn.Embedding(n_entities, dim)
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, i, j):
        ei = self.embed(i)
        ej = self.embed(j)
        x = torch.cat([ei, ej], dim=-1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)

model = EntityEncoder(n_entities)
opt = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# ── Train ────────────────────────────────────────────────────────────────
print("Training link predictor...")
batch_size = 64
for epoch in range(50):
    perm = torch.randperm(len(train_e))
    total_loss = 0
    for start in range(0, len(train_e), batch_size):
        idx = perm[start:start + batch_size]
        i, j = train_e[idx, 0], train_e[idx, 1]
        pred = model(i, j)
        loss = loss_fn(pred, train_l[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if epoch % 10 == 0 or epoch == 49:
        with torch.no_grad():
            preds = model(test_e[:, 0], test_e[:, 1])
            acc = ((preds > 0.5) == test_l.bool()).float().mean().item()
        print(f"  E{epoch:3d} loss={total_loss:.4f} test_acc={acc:.3f}")

# ── Query with embeddings ────────────────────────────────────────────────
def find_similar(entity, top_k=10):
    if entity not in e2i:
        return []
    idx = e2i[entity]
    with torch.no_grad():
        emb = model.embed.weight
        sims = (emb[idx] @ emb.T).softmax(dim=-1)
        vals, ids = sims.topk(top_k + 1)
    return [(entity_names[ids[k].item()], vals[k].item()) for k in range(1, top_k + 1)]

def predict_links(entity, top_k=10):
    if entity not in e2i:
        return []
    idx = e2i[entity]
    with torch.no_grad():
        all_i = torch.full((n_entities,), idx, dtype=torch.long)
        all_j = torch.arange(n_entities)
        preds = model(all_i, all_j)
        vals, ids = preds.topk(top_k + 1)
    return [(entity_names[ids[k].item()], vals[k].item()) for k in range(1, top_k + 1)]

# ── Demo ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Similar entities (embedding cosine):")
test = entity_names[min(5, n_entities - 1)]
print(f"  {test}:")
for name, sim in find_similar(test, 5):
    print(f"    {name}: {sim:.3f}")

print(f"\nPredicted links for '{test}':")
for name, prob in predict_links(test, 5):
    print(f"    {name}: {prob:.3f}")

# Save embeddings for downstream use
torch.save(model.embed.weight, "entity_embeddings.pt")
print("\nSaved: entity_embeddings.pt")
