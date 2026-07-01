"""graficar.py — Load .pkl KG and generate PNG visualizations."""
import os, sys, pickle, math
from collections import defaultdict

os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
except ImportError:
    os.system(f"{sys.executable} -m pip install matplotlib networkx -q")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

CACHE_FILE = "knowledge_graph.pkl"
OUTPUT = "graf1.png"

if not os.path.exists(CACHE_FILE):
    print(f"Run procesar.py first — {CACHE_FILE} not found")
    sys.exit(1)

with open(CACHE_FILE, "rb") as f:
    kg = pickle.load(f)

entity_names = kg["entity_names"]
cooc = kg["cooc"]
hierarchy = kg["hierarchy"]
cluster_names = kg["cluster_names"]
n_clusters = kg["n_clusters"]


def draw(output_path, title="Knowledge Graph"):
    G = nx.Graph()
    for n in entity_names: G.add_node(n)
    for (a, b), w in cooc.items():
        G.add_edge(a, b, weight=w)

    pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)
    colors = ["#e74c3c","#2ecc71","#3498db","#f1c40f","#9b59b6",
              "#1abc9c","#e67e22","#2980b9","#e84393","#00b894"]

    # Color by category
    nc = {}
    for i, (cid, members) in enumerate(hierarchy.items()):
        c = colors[i % len(colors)]
        for n in members: nc[n] = c

    fig, ax = plt.subplots(1, 1, figsize=(24, 18), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    max_w = max(cooc.values()) if cooc else 1
    for (a, b), w in cooc.items():
        if a in pos and b in pos:
            lw = 0.2 + 2.5 * w / max_w
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color="#495057", lw=lw, alpha=0.3 + 0.5 * w / max_w, zorder=1)

    deg = dict(G.degree())
    max_d = max(deg.values()) if deg else 1
    for n, (x, y) in pos.items():
        r = 0.02 + 0.12 * deg.get(n, 1) / max_d
        c = nc.get(n, "#868e96")
        circle = plt.Circle((x, y), r, color=c, ec="white", lw=1.2, alpha=0.85, zorder=5)
        ax.add_patch(circle)
        label = n.replace("_", " ").title()
        fs = 4 + 5 * deg.get(n, 1) / max_d
        ax.text(x, y + r + 0.02, label, ha="center", va="bottom",
                fontsize=fs, color="white", fontweight="bold", zorder=6)

    legend = [
        plt.Line2D([0],[0], color="#495057", lw=3, label="Co-occurrence"),
        plt.Circle((0,0), 0.06, color="#3498db", label="Categories"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8,
              framealpha=0.7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax.text(0.5, 0.98, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=14, color="white", fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved: {output_path}")


def draw_category(cat_name, output_path="category.png"):
    """Draw subgraph of a single category."""
    cat_entities = []
    for cid, members in hierarchy.items():
        cname = cluster_names.get(cid, cid)
        if cat_name.lower() in cname.lower():
            cat_entities.extend(members)
    if not cat_entities:
        print(f"Category '{cat_name}' not found")
        return
    # Build subgraph
    G = nx.Graph()
    for n in cat_entities: G.add_node(n)
    for (a, b), w in cooc.items():
        if a in cat_entities and b in cat_entities:
            G.add_edge(a, b, weight=w)
    if not G.edges():
        # Add weak connections
        for i in range(len(cat_entities) - 1):
            G.add_edge(cat_entities[i], cat_entities[i+1], weight=1)

    pos = nx.spring_layout(G, k=1.5, iterations=200, seed=42)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for a, b, d in G.edges(data=True):
        if a in pos and b in pos:
            w = d.get("weight", 1)
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color="#495057", lw=0.5 + 2 * w / max(1, max(d.get("weight", 1) for _, _, d in G.edges(data=True))),
                    alpha=0.5, zorder=1)
    deg = dict(G.degree())
    max_d = max(deg.values()) if deg else 1
    for n, (x, y) in pos.items():
        r = 0.04 + 0.10 * deg.get(n, 1) / max_d
        circle = plt.Circle((x, y), r, color="#3498db", ec="white", lw=1.2, alpha=0.85, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y + r + 0.02, n.replace("_", " ").title(), ha="center", va="bottom",
                fontsize=6, color="white", fontweight="bold", zorder=6)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved category graph: {output_path}")


if __name__ == "__main__":
    draw(OUTPUT)
    for cid, cname in list(cluster_names.items())[:3]:
        draw_category(cname, f"{cname.replace(' ', '_')}.png")
