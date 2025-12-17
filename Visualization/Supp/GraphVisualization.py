import matplotlib.pyplot as plt
import networkx as nx

# ---------- Graph definition ----------
G = nx.DiGraph()
G.add_nodes_from([1, 2, 3, 4, 5])

# (u, v, weight/label, curvature)
edges = [
    (1, 2, "A2",  0.25),
    (2, 1, "A1", -0.25),
    (2, 1, "A1", -0.25),
    (2, 3, "B",   0.00),
    (3, 1, "C",   0.00),
    (3, 4, "D2",  0.25),
    (4, 3, "D1", -0.25),
    (1, 5, "E1",  0.25),
    (5, 1, "E2", -0.25),
    (3, 5, "F",   0.00),
    (4, 5, "G",   0.00),
]

for u, v, w, rad in edges:
    G.add_edge(u, v, weight=w, rad=rad)

# Fixed positions to mimic your layout
pos = {
    1: (0.0, 0.0),
    2: (-1.2, -0.2),
    3: (0.0, 0.9),
    4: (0.0, 1.7),
    5: (1.2, 0.9),
}

# ---------- Drawing ----------
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.set_axis_off()

# Nodes
nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#1f78b4",
                       edgecolors="black", linewidths=1.0, ax=ax)
nx.draw_networkx_labels(G, pos, font_color="white",
                        font_weight="bold", ax=ax)

# Edges with individual curvature
for u, v, data in G.edges(data=True):
    rad = data["rad"]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v)],
        arrowstyle="-|>",
        arrowsize=15,
        connectionstyle=f"arc3,rad={rad}",
        ax=ax,
    )

# Edge labels = weights
edge_labels = {(u, v): data["weight"] for u, v, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=9,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
    ax=ax,
)

plt.tight_layout()
plt.show()
