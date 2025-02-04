import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import string

results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\results\\bc\\"

def normalize(mat):
    max_score = np.max(mat)
    mat /= max_score
    mat = 1 - mat
    _, _, t = mat.shape
    for i in range(t):
        A = mat[:, :, i]
        np.fill_diagonal(A, 0)
        mat[:, :, i] = A
    return mat


adj_mat = np.load(results_path + "adj_mat.npy")
adj_mat = normalize(adj_mat)
color_map = ["#8dd3c7", "#b3de69", "#bebada", "#fb8072", "#80b1d3", "#fdb462"]
fig, ax = plt.subplots(2, 5)
for i in range(10):
    A = adj_mat[:, :, i]
    G = nx.from_numpy_array(A)

    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))

    ix = np.unravel_index(i, ax.shape)
    plt.sca(ax[ix])
    my_pos = nx.spring_layout(G, seed=2)
    nx.draw(G, pos=my_pos, with_labels=True, ax=ax[ix], node_color=color_map)

    ax[ix].set_title("Min="+str(i+1), fontsize=10)
    ax[ix].set_axis_off()

plt.show()
# print(adj_mat)