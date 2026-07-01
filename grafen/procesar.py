"""procesar.py — Download text, extract entities, build KG, save .pkl"""
import os, sys, json, re, pickle
from collections import defaultdict, Counter

# ── Install deps ─────────────────────────────────────────────────────────
os.system(f"{sys.executable} -m pip install spacy scikit-learn sentence-transformers datasets -q")
os.system(f"{sys.executable} -m spacy download es_core_news_sm -q")

import spacy
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────
DATA_SOURCE = "wiki"         # "wiki" or "file"
FILE_PATH = "../rust/input.txt"
MAX_ARTICLES = 500
MAX_ENTITIES = 200
MAX_MB = 100
CACHE_FILE = "knowledge_graph.pkl"
WIKI_CACHE = "wiki_100mb.json"

# ── Load / Download text ─────────────────────────────────────────────────
def load_texts():
    if os.path.exists(WIKI_CACHE):
        with open(WIKI_CACHE, encoding="utf-8") as f:
            texts = json.load(f)
        print(f"Loaded {len(texts)} articles from {WIKI_CACHE}")
        return texts
    if DATA_SOURCE == "file":
        with open(FILE_PATH, encoding="utf-8") as f:
            return [f.read()]
    print("Downloading Wikipedia ES...")
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    texts, total = [], 0
    for item in ds:
        t = f"{item['title']}. {item['text']}"
        texts.append(t)
        total += len(t.encode("utf-8"))
        if total >= MAX_MB * 1024 * 1024:
            break
    with open(WIKI_CACHE, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    print(f"Saved {len(texts)} articles ({total//1024//1024}MB) to {WIKI_CACHE}")
    return texts

# ── Build KG ─────────────────────────────────────────────────────────────
def build_kg(texts):
    print("Loading spaCy NER...")
    nlp = spacy.load("es_core_news_sm")
    bert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Extract entities
    print("Extracting entities...")
    entities = {}
    for text in texts[:MAX_ARTICLES]:
        doc = nlp(text[:10000])
        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < 3: continue
            if name not in entities:
                entities[name] = {"type": ent.label_, "count": 0, "contexts": []}
            entities[name]["count"] += 1
            if len(entities[name]["contexts"]) < 5:
                entities[name]["contexts"].append(text[max(0, ent.start_char-50):ent.end_char+50])

    entity_names = list(entities.keys())[:MAX_ENTITIES]
    entity_counts = [entities[e]["count"] for e in entity_names]
    print(f"  {len(entity_names)} entities")

    # Semantic clusters
    print("Clustering entities...")
    embeds = bert.encode(entity_names, show_progress_bar=True)
    n_clusters = min(15, max(3, len(entity_names) // 5))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeds)

    hierarchy = defaultdict(list)
    for name, cid in zip(entity_names, clusters):
        hierarchy[f"cat_{cid}"].append(name)

    cluster_names = {}
    for i in range(n_clusters):
        members = [(entity_names[j], entity_counts[j]) for j in range(len(entity_names)) if clusters[j] == i]
        members.sort(key=lambda x: -x[1])
        cluster_names[f"cat_{i}"] = members[0][0] if members else f"cat_{i}"

    # Co-occurrence relationships
    print("Building relationships...")
    cooc = defaultdict(int)
    entity_set = set(entity_names)
    for text in texts[:MAX_ARTICLES]:
        doc = nlp(text[:10000])
        found = set(ent.text for ent in doc.ents if ent.text in entity_set)
        for e1 in found:
            for e2 in found:
                if e1 < e2:
                    cooc[tuple(sorted([e1, e2]))] += 1

    kg = {
        "entities": entities,
        "cooc": dict(cooc),
        "hierarchy": dict(hierarchy),
        "cluster_names": cluster_names,
        "entity_names": entity_names,
        "embeds": embeds,
        "n_clusters": n_clusters,
    }
    return kg

# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    texts = load_texts()
    kg = build_kg(texts)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(kg, f)
    print(f"Saved KG to {CACHE_FILE}")
    print(f"  Entities: {len(kg['entity_names'])}")
    print(f"  Relationships: {len(kg['cooc'])}")
    print(f"  Categories: {kg['n_clusters']}")
