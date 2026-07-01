"""consultar.py — Load .pkl KG and query interactively (no transformers)."""
import os, sys, pickle
from collections import defaultdict

CACHE_FILE = "knowledge_graph.pkl"

if not os.path.exists(CACHE_FILE):
    print(f"Run procesar.py first — {CACHE_FILE} not found")
    sys.exit(1)

with open(CACHE_FILE, "rb") as f:
    kg = pickle.load(f)

entities = kg["entities"]
cooc = kg["cooc"]
hierarchy = kg["hierarchy"]
cluster_names = kg["cluster_names"]
entity_names = kg["entity_names"]


def query(name):
    name_lower = name.lower()
    matches = [e for e in entity_names if name_lower in e.lower()]
    if not matches:
        # Try semantic fallback if embeddings exist
        if "embeds" in kg:
            import numpy as np
            q_emb = np.zeros((1, kg["embeds"].shape[1]))
            # Simple word overlap fallback
            words = name_lower.split()
            for e in entity_names:
                if any(w in e.lower() for w in words):
                    matches.append(e)
        if not matches:
            return f"No match for '{name}'"
        matches = matches[:5]

    results = {}
    for ent in matches[:3]:
        info = entities.get(ent, {})
        relations = []
        for (a, b), w in sorted(cooc.items(), key=lambda x: -x[1])[:100]:
            if a == ent: relations.append((b, w))
            elif b == ent: relations.append((a, w))
        relations = relations[:15]

        cat = None
        for cid, members in hierarchy.items():
            if ent in members:
                cat = cluster_names.get(cid, cid)
                break
        same_cat = []
        if cat:
            for cid, cname in cluster_names.items():
                if cname == cat:
                    same_cat = [m for m in hierarchy[cid] if m != ent][:10]
                    break
        results[ent] = {
            "type": info.get("type", "?"),
            "category": cat,
            "count": info.get("count", 0),
            "relations": relations,
            "same_category": same_cat,
            "contexts": info.get("contexts", [])[:2],
        }
    return results


def list_category(cat_name):
    for cid, cname in cluster_names.items():
        if cat_name.lower() in cname.lower() or cat_name.lower() in cid.lower():
            return hierarchy[cid]
    results = []
    for cid, members in hierarchy.items():
        base = cluster_names.get(cid, cid)
        if cat_name.lower() in base.lower():
            results.extend(members)
    return results[:30]


def export(query_name):
    data = query(query_name)
    if isinstance(data, str):
        return []
    all_ents = set()
    for ent, info in data.items():
        all_ents.add(ent)
        for r, _ in info.get("relations", []):
            all_ents.add(r)
        for r in info.get("same_category", []):
            all_ents.add(r)
    return [{"entity": e, "type": entities.get(e, {}).get("type", "?"),
             "count": entities.get(e, {}).get("count", 0)} for e in all_ents]


if __name__ == "__main__":
    print(f"Knowledge Graph loaded: {len(entity_names)} entities, {len(cooc)} relationships")
    print(f"Categories: {len(cluster_names)}")
    for cid, cname in sorted(cluster_names.items()):
        print(f"  {cname}: {len(hierarchy[cid])} entities")

    print("\nCommands: <query> | cat:<name> | export:<name> | exit")
    while True:
        try:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q.startswith("cat:"):
                items = list_category(q[4:].strip())
                print(", ".join(items[:25]))
            elif q.startswith("export:"):
                items = export(q[7:].strip())
                print(json.dumps(items, indent=2, ensure_ascii=False))
            else:
                r = query(q)
                if isinstance(r, str):
                    print(r)
                else:
                    for ent, info in r.items():
                        print(f"\n  [{info['type']}] {ent} (x{info['count']})")
                        print(f"    Category: {info['category']}")
                        print(f"    Relations: {', '.join(f'{rel}({w})' for rel,w in info['relations'][:8])}")
                        print(f"    Same cat: {', '.join(info['same_category'][:6])}")
        except (KeyboardInterrupt, EOFError):
            break
