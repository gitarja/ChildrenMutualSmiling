import networkx as nx
import numpy as np
import string
color_map = ["#8dd3c7", "#b3de69", "#bebada", "#fb8072", "#80b1d3"]
import matplotlib.pyplot as plt
def plotGraph(A, save=False, filename=""):
    A[A> 0.5] = A[A> 0.5] * 5
    #A = 1- A

    np.fill_diagonal(A, 0)
    #print(A)
    G = nx.from_numpy_array(A)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), string.ascii_uppercase)))

    my_pos = nx.spring_layout(G, seed=25, iterations=1000, k=10, threshold=1e-9, scale=10, fixed=None, center=np.zeros((2)))

    nx.draw(G, pos=my_pos, with_labels=True,  node_color=color_map, node_size=1000)

    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()
