import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import numpy as np
import networkx as nx

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from models.gnn_model import GCN
from utils.visualization import visualize_graph, plot_feature_distribution


# =====================================
# LOAD DATASET
# =====================================

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target


# =====================================
# NORMALIZE FEATURES
# =====================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)


# =====================================
# CREATE GRAPH
# =====================================

A = kneighbors_graph(
    X,
    n_neighbors=5,
    mode='connectivity',
    include_self=False
)

edge_index = torch.tensor(
    np.array(A.nonzero()),
    dtype=torch.long
)


data_graph = Data(
    x=X,
    edge_index=edge_index,
    y=y
)


# =====================================
# VISUALIZATIONS
# =====================================

G = nx.Graph()

edges = edge_index.t().tolist()
G.add_edges_from(edges)

visualize_graph(G)
plot_feature_distribution(X)

# =====================================
# TRAIN TEST SPLIT
# =====================================

num_nodes = data_graph.num_nodes
indices = np.random.permutation(num_nodes)

train_count = int(0.7 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[indices[:train_count]] = True
test_mask[indices[train_count:]] = True


data_graph.train_mask = train_mask
data_graph.test_mask = test_mask


# =====================================
# MODEL
# =====================================

model = GCN(
    input_dim=X.shape[1],
    hidden_dim=16,
    output_dim=2
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# =====================================
# TRAIN FUNCTION
# =====================================


def train():
    model.train()

    optimizer.zero_grad()

    out = model(data_graph.x, data_graph.edge_index)

    loss = F.cross_entropy(
        out[data_graph.train_mask],
        data_graph.y[data_graph.train_mask]
    )

    loss.backward()
    optimizer.step()

    return loss.item()


# =====================================
# TEST FUNCTION
# =====================================


def test():
    model.eval()

    with torch.no_grad():
        pred = model(data_graph.x, data_graph.edge_index).argmax(dim=1)

        correct = (
            pred[data_graph.test_mask]
            == data_graph.y[data_graph.test_mask]
        ).sum()

        acc = int(correct) / int(data_graph.test_mask.sum())

    return acc


# =====================================
# TRAIN MODEL
# =====================================

print("Training GNN Model...\n")

for epoch in range(1, 101):
    loss = train()

    if epoch % 10 == 0:
        acc = test()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")


# =====================================
# FINAL ACCURACY
# =====================================

final_acc = test()

print("\n============================")
print(f"Final Test Accuracy: {final_acc:.4f}")
print("============================")


# =====================================
# t-SNE VISUALIZATION
# =====================================

model.eval()

with torch.no_grad():
    h = model.conv1(data_graph.x, data_graph.edge_index)
    h = h.detach().cpu().numpy()


tsne = TSNE(
    n_components=2,
    random_state=0,
    perplexity=30
)

z = tsne.fit_transform(h)

plt.figure(figsize=(8, 6))

plt.scatter(
    z[:, 0],
    z[:, 1],
    c=y,
)

plt.title("t-SNE Visualization of Node Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.show()