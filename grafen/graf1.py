"""Knowledge Graph — Colab: download Wikipedia ES, extract entities, query."""
# ── Install ──────────────────────────────────────────────────────────────
import subprocess, sys, os, json, re, math
from collections import defaultdict, Counter

subprocess.run([sys.executable, "-m", "pip", "install", "spacy", "scikit-learn", "sentence-transformers", "datasets", "-q"], capture_output=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm", "-q"], capture_output=True)
import spacy
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# ── Download Wikipedia ES 100MB ──────────────────────────────────────────
print("Downloading ~100MB Spanish Wikipedia...")
from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)

texts = []
total = 0
for item in ds:
    t = f"{item['title']}. {item['text']}"
    texts.append(t)
    total += len(t.encode("utf-8"))
    if total >= 100_000_000:
        break
print(f"Downloaded {len(texts)} articles, ~{total//1024//1024}MB")

# ── Load models ──────────────────────────────────────────────────────────
print("Loading NLP models...")
nlp = spacy.load("es_core_news_sm")
bert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ── Extract entities ─────────────────────────────────────────────────────
print("Extracting entities (NER)...")
entities = {}  # name -> {type, count, contexts}
for text in texts[:500]:  # sample first 500 articles
    doc = nlp(text[:10000])
    for ent in doc.ents:
        name = ent.text.strip()
        if len(name) < 3:
            continue
        if name not in entities:
            entities[name] = {"type": ent.label_, "count": 0, "contexts": [], "articles": set()}
        entities[name]["count"] += 1
        if len(entities[name]["contexts"]) < 5:
            entities[name]["contexts"].append(doc.text[max(0, ent.start_char-50):ent.end_char+50])
        entities[name]["articles"].add(text[:50])

print(f"Found {len(entities)} entities")

# ── Build knowledge graph ────────────────────────────────────────────────
print("Building knowledge graph...")
entity_names = list(entities.keys())[:200]  # top 200
entity_types = [entities[e]["type"] for e in entity_names]
entity_counts = [entities[e]["count"] for e in entity_names]

# Embeddings for semantic clustering
embeds = bert.encode(entity_names, show_progress_bar=True)

# Cluster into categories
n_clusters = min(15, len(entity_names) // 5)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeds)

# Build hierarchy: cluster → entity
hierarchy = defaultdict(list)
for name, cluster_id in zip(entity_names, clusters):
    hierarchy[f"category_{cluster_id}"].append(name)

# Name categories by most representative entity
cluster_names = {}
for i in range(n_clusters):
    members = [(entity_names[j], entity_counts[j]) for j in range(len(entity_names)) if clusters[j] == i]
    members.sort(key=lambda x: -x[1])
    cluster_names[f"category_{i}"] = members[0][0] if members else f"cat_{i}"

# Co-occurrence edges (within same article)
print("Building relationships...")
cooc = defaultdict(int)
entity_set = set(entity_names)
article_entities = []
for text in texts[:500]:
    doc = nlp(text[:10000])
    found = set()
    for ent in doc.ents:
        if ent.text in entity_set:
            found.add(ent.text)
    article_entities.append(found)
    for e1 in found:
        for e2 in found:
            if e1 < e2:
                cooc[tuple(sorted([e1, e2]))] += 1

# ── Query engine ─────────────────────────────────────────────────────────
class KnowledgeGraph:
    def __init__(self, entities, cooc, hierarchy, cluster_names, embeds, entity_names):
        self.entities = entities
        self.cooc = cooc
        self.hierarchy = hierarchy
        self.cluster_names = cluster_names
        self.embeds = embeds
        self.entity_names = entity_names
        self.name_to_idx = {n: i for i, n in enumerate(entity_names)}

    def query(self, name, depth=2):
        """Query an entity: relationships, category, contexts."""
        name_lower = name.lower()
        # Find closest match
        matches = [e for e in self.entity_names if name_lower in e.lower()]
        if not matches:
            # Semantic fallback: find closest by embedding
            q_emb = bert.encode([name])
            sims = (self.embeds @ q_emb.T).flatten()
            best = sims.argsort()[-5:][::-1]
            matches = [self.entity_names[i] for i in best if sims[i] > 0.3]
            if not matches:
                return f"No match for '{name}'"

        result = {}
        for ent in matches[:3]:
            info = self.entities.get(ent, {})
            # Find relationships
            relations = []
            for (a, b), w in self.cooc.most_common(100):
                if a == ent:
                    relations.append((b, w))
                elif b == ent:
                    relations.append((a, w))
            relations.sort(key=lambda x: -x[1])
            # Find category
            cat = None
            for cid, members in self.hierarchy.items():
                if ent in members:
                    cat = self.cluster_names.get(cid, cid)
                    break
            # Find same-category entities
            same_cat = []
            if cat:
                for cid, cname in self.cluster_names.items():
                    if cname == cat:
                        same_cat = [m for m in self.hierarchy[cid] if m != ent][:10]
                        break
            result[ent] = {
                "type": info.get("type", "?"),
                "category": cat,
                "count": info.get("count", 0),
                "relations": relations[:15],
                "same_category": same_cat,
                "contexts": info.get("contexts", [])[:3],
            }
        return result

    def list_category(self, category_name):
        """List all entities in a category."""
        for cid, cname in self.cluster_names.items():
            if category_name.lower() in cname.lower() or category_name.lower() in cid.lower():
                return self.hierarchy[cid]
        # Search across categories
        results = []
        for cid, members in self.hierarchy.items():
            base = self.cluster_names.get(cid, cid)
            if category_name.lower() in base.lower():
                results.extend(members)
        return results[:30]

    def export(self, query_name):
        """Export all related entities for training."""
        data = self.query(query_name)
        all_entities = set()
        for ent, info in data.items():
            all_entities.add(ent)
            for rel, _ in info.get("relations", []):
                all_entities.add(rel)
            for rel in info.get("same_category", []):
                all_entities.add(rel)
        # Get contexts
        export = []
        for ent in all_entities:
            info = self.entities.get(ent, {})
            export.append({
                "entity": ent,
                "type": info.get("type", "?"),
                "count": info.get("count", 0),
                "contexts": info.get("contexts", [])[:2],
            })
        return export

kg = KnowledgeGraph(entities, cooc, hierarchy, cluster_names, embeds, entity_names)

# ── Demo ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("KNOWLEDGE GRAPH READY — Query examples:")
print("="*60)
print("\nCategories found:")
for cid, cname in sorted(cluster_names.items()):
    n_members = len(hierarchy[cid])
    print(f"  {cname}: {n_members} entities")

# Test query
test = "Romeo" if "Romeo" in entity_names else entity_names[0]
print(f"\nQuery: '{test}'")
result = kg.query(test)
for ent, info in result.items():
    print(f"\n  [{info['type']}] {ent} (cat: {info['category']})")
    print(f"    Relations: {', '.join(r[0] for r in info['relations'][:8])}")
    print(f"    Same cat: {', '.join(info['same_category'][:5])}")

print("\n" + "="*60)
print("Interactive: type 'exit' to quit")
while True:
    try:
        q = input("\nQuery > ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if q.startswith("cat:"):
            items = kg.list_category(q[4:].strip())
            print(f"Entities in '{q[4:].strip()}': {', '.join(items[:20])}")
        else:
            r = kg.query(q)
            if isinstance(r, str):
                print(r)
            else:
                for ent, info in r.items():
                    print(f"\n  [{info['type']}] {ent} (x{info['count']})")
                    print(f"    Category: {info['category']}")
                    print(f"    Relations: {', '.join(f'{rel}({w})' for rel,w in info['relations'][:10])}")
                    print(f"    Same cat: {', '.join(info['same_category'][:8])}")
    except KeyboardInterrupt:
        break

# ── Save for reuse ───────────────────────────────────────────────────────
import pickle
with open("knowledge_graph.pkl", "wb") as f:
    pickle.dump({"entities": entities, "cooc": cooc, "hierarchy": hierarchy,
                 "cluster_names": cluster_names, "entity_names": entity_names}, f)
print("\nSaved: knowledge_graph.pkl")
