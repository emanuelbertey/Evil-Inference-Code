"""Generate a PNG visualization of recurrent knowledge graph nodes."""
import os, sys, math, random
from collections import defaultdict

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None
    mpatches = None

try:
    import networkx as nx
    HAVE_NX = True
except ImportError:
    nx = None
    HAVE_NX = False

_DIR = os.path.dirname(os.path.abspath(__file__))


def build_knowledge_graph(input_path):
    """Parse input into a DAG with recurrent edges."""
    G = nx.DiGraph() if HAVE_NX else None
    tokens = set()
    edges = []
    with open(input_path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            for i in range(len(parts) - 1):
                edges.append((parts[i], parts[i + 1]))
                tokens.add(parts[i])
                tokens.add(parts[i + 1])
            if len(parts) > 0:
                tokens.add(parts[0])
    tokens = list(tokens)
    if HAVE_NX:
        for t in tokens:
            G.add_node(t)
        for a, b in edges:
            G.add_edge(a, b)
    return G, tokens, edges


def build_recurrent_clusters(tokens, edges):
    """Build clusters from connected components and add recurrent edges."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    clusters = []
    for t in tokens:
        if t in visited:
            continue
        stack = [t]
        comp = set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in adj[n]:
                if nb not in visited:
                    stack.append(nb)
        clusters.append(list(comp))
    # Add recurrent self-loop edges
    rec_edges = []
    for i in range(len(tokens)):
        rec_edges.append((tokens[i], tokens[(i + 1) % len(tokens)]))
    return clusters, rec_edges


def draw_graph(G, tokens, edges, clusters, rec_edges, output_path):
    if not HAVE_NX:
        draw_graph_manual(tokens, edges, clusters, rec_edges, output_path)
        return
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    fig, ax = plt.subplots(1, 1, figsize=(20, 14), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    # Cluster colors
    colors = ["#ff6b6b", "#51cf66", "#5c7cfa", "#fcc419", "#cc5de8",
              "#20c997", "#ff922b", "#748ffc", "#f06595", "#38d9a9"]
    node_colors = {}
    for i, cl in enumerate(clusters):
        c = colors[i % len(colors)]
        for n in cl:
            node_colors[n] = c
    # Edges
    for a, b in edges:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="->", color="#495057",
                                        lw=1.2, alpha=0.6, connectionstyle="arc3,rad=0.1"))
    # Recurrent edges (dashed, colored)
    for a, b in rec_edges:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="->", color="#fcc419",
                                        lw=0.8, alpha=0.4, connectionstyle="arc3,rad=0.3",
                                        linestyle="dashed"))
    # Nodes
    for n, (x, y) in pos.items():
        c = node_colors.get(n, "#868e96")
        circle = plt.Circle((x, y), 0.12, color=c, ec="white", lw=1.5, alpha=0.9, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y + 0.16, n.replace("_", " ").title(), ha="center", va="bottom",
                fontsize=7, color="white", fontweight="bold", zorder=6)
    # Legend
    legend_elements = [
        mpatches.Patch(color="#495057", label="Feedforward edge"),
        plt.Line2D([0], [0], color="#fcc419", lw=1.5, linestyle="--", label="Recurrent edge"),
        plt.Circle((0, 0), 0.05, color="#868e96", label="Knowledge node"),
    ]
    for i, cl in enumerate(clusters):
        if i < 6:
            c = colors[i % len(colors)]
            name = f"Cluster {i+1}"
            legend_elements.append(plt.Circle((0, 0), 0.05, color=c, label=name))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
              framealpha=0.7, facecolor="#1a1a2e", edgecolor="white",
              labelcolor="white")
    # Title
    ax.text(0.5, 0.98, "Recurrent Knowledge Graph — Recurrence Nodes & Knowledge Trees",
            transform=ax.transAxes, ha="center", va="top", fontsize=14,
            color="white", fontweight="bold")
    ax.text(0.5, 0.94, "Solid = feedforward propagation | Dashed = recurrent cycles | Colors = knowledge clusters",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#adb5bd")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


def draw_graph_manual(tokens, edges, clusters, rec_edges, output_path):
    """Fallback manual graph draw using matplotlib only."""
    n = len(tokens)
    angles = [2 * math.pi * i / max(n, 1) - math.pi / 2 for i in range(n)]
    radius = 0.4
    pos = {t: (radius * math.cos(a), radius * math.sin(a)) for t, a in zip(tokens, angles)}
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    colors = ["#ff6b6b", "#51cf66", "#5c7cfa", "#fcc419", "#cc5de8",
              "#20c997", "#ff922b", "#748ffc", "#f06595", "#38d9a9"]
    node_colors = {}
    for i, cl in enumerate(clusters):
        c = colors[i % len(colors)]
        for nn in cl:
            node_colors[nn] = c
    # Edges
    for a, b in edges:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="->", color="#495057",
                                        lw=1.2, alpha=0.6, connectionstyle="arc3,rad=0.1"))
    # Recurrent edges
    for a, b in rec_edges:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="->", color="#fcc419",
                                        lw=0.8, alpha=0.4, connectionstyle="arc3,rad=0.3",
                                        linestyle="dashed"))
    # Nodes
    for nn, (x, y) in pos.items():
        c = node_colors.get(nn, "#868e96")
        circle = plt.Circle((x, y), 0.08, color=c, ec="white", lw=1.5, alpha=0.9, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y + 0.1, nn.replace("_", " ").title(), ha="center", va="bottom",
                fontsize=6, color="white", fontweight="bold", zorder=6)
    legend_elements = [
        mpatches.Patch(color="#495057", label="Feedforward"),
        plt.Line2D([0], [0], color="#fcc419", lw=1.5, linestyle="--", label="Recurrent"),
        plt.Circle((0, 0), 0.05, color="#868e96", label="Node"),
    ]
    for i, cl in enumerate(clusters):
        if i < 6:
            c = colors[i % len(colors)]
            legend_elements.append(plt.Circle((0, 0), 0.05, color=c, label=f"C{i+1}"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              framealpha=0.7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax.text(0.5, 0.98, "Recurrent Knowledge Graph", transform=ax.transAxes,
            ha="center", va="top", fontsize=14, color="white", fontweight="bold")
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


def main():
    input_path = os.path.join(_DIR, "rust", "input.txt")
    output_path = os.path.join(_DIR, "graf1.png")
    if not os.path.exists(input_path):
        print(f"input.txt not found at {input_path}")
        sys.exit(1)
    G, tokens, edges = build_knowledge_graph(input_path)
    clusters, rec_edges = build_recurrent_clusters(tokens, edges)
    draw_graph(G, tokens, edges, clusters, rec_edges, output_path)
    print(f"Graph saved: {output_path}")
    print(f"  Nodes: {len(tokens)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Recurrent edges: {len(rec_edges)}")
    print(f"  Clusters: {len(clusters)}")


if __name__ == "__main__":
    main()
