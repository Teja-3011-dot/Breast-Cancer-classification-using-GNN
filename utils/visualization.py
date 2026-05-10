import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np


def visualize_graph(G):
    random.seed(0)
    np.random.seed(0)

    sample_nodes = random.sample(list(G.nodes()), 50)
    H = G.subgraph(sample_nodes).copy()

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(H, seed=0)

    nx.draw(
        H,
        pos,
        node_size=80,
        with_labels=False
    )

    plt.title("Sample Graph Visualization")
    plt.show()



def plot_feature_distribution(X):
    features_flat = X.numpy().flatten()

    plt.hist(features_flat, bins=50)
    plt.title("Node Feature Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.show()