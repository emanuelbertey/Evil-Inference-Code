"""filter.py — Filter Wikipedia articles by topic relevance using KG + embeddings."""
import os, sys, json, pickle, re
from collections import defaultdict

os.system(f"{sys.executable} -m pip install sentence-transformers -q")
from sentence_transformers import SentenceTransformer

CACHE_FILE = "knowledge_graph.pkl"
WIKI_FILE = "wiki_100mb.json"

if not os.path.exists(WIKI_FILE):
    print("Run procesar.py first to download Wikipedia")
    sys.exit(1)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        kg = pickle.load(f)

print("Loading embedding model...")
bert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with open(WIKI_FILE, encoding="utf-8") as f:
    articles = json.load(f)
print(f"Loaded {len(articles)} articles")

# ── Filter by topic ──────────────────────────────────────────────────────
def filter_by_topic(topic, threshold=0.25, max_results=None):
    """Filter articles by semantic relevance to topic."""
    topic_emb = bert.encode([topic])
    # Process in batches
    batch_size = 64
    results = []
    for start in range(0, len(articles), batch_size):
        batch = articles[start:start + batch_size]
        # Truncate for speed: first 500 chars
        texts = [a[:500] for a in batch]
        embs = bert.encode(texts, show_progress_bar=False)
        sims = (embs @ topic_emb.T).flatten()
        for i, sim in enumerate(sims):
            if sim >= threshold:
                results.append((batch[i], sim.item()))
    results.sort(key=lambda x: -x[1])
    if max_results:
        results = results[:max_results]
    return results

# ── Export ───────────────────────────────────────────────────────────────
def export_filtered(topic, output_file):
    filtered = filter_by_topic(topic)
    with open(output_file, "w", encoding="utf-8") as f:
        for text, score in filtered:
            # Extract title (first line before period)
            title = text.split(".")[0] if "." in text else text[:80]
            f.write(f"# [{score:.2f}] {title}\n")
            f.write(text + "\n\n")
    print(f"Saved {len(filtered)} articles to {output_file}")
    return filtered

# ── Interactive ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("filter", exist_ok=True)
    print("\nFilter Wikipedia by topic. Examples: fútbol, medicina, ciencia, historia")
    print("Commands: <topic> | export:<topic> | exit")
    while True:
        try:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q.startswith("export:"):
                topic = q[7:].strip()
                out = f"filter/{topic.replace(' ', '_')}.txt"
                filtered = export_filtered(topic, out)
                avg_score = sum(s for _, s in filtered) / len(filtered) if filtered else 0
                print(f"  Avg relevance: {avg_score:.2f} | Articles: {len(filtered)}")
            else:
                results = filter_by_topic(q, max_results=10)
                if not results:
                    print("  No matches, try lowering threshold or different topic")
                for text, score in results[:10]:
                    title = text.split(".")[0] if "." in text else text[:80]
                    print(f"  [{score:.2f}] {title[:120]}")
        except (KeyboardInterrupt, EOFError):
            break
