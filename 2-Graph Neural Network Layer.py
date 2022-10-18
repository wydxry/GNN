import torch
import numpy as np
from torch_geometric.datasets import KarateClub
# Visualization libraries
import matplotlib.pyplot as plt
import networkx as nx

# Import dataset from PyTorch Geometric
dataset = KarateClub()

# Print information
print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Graph: Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
print(f'Graph: {dataset[0]}')

data = dataset[0]

# Print x
print(f'x = {data.x.shape}')
print(data.x)

print(f'edge_index = {data.edge_index.shape}')
print(data.edge_index)
# print(data.edge_index[0])
# print(data.edge_index[1])

from torch_geometric.utils import to_dense_adj

# The adjacency matrix can actually be calculated from the edge_index with a utility function.
A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print(f'A = {A.shape}')
print(A)

print(f'y = {data.y.shape}')
print(data.y)

# the train_mask shows which nodes are supposed to be used for training with True statements.
print(f'train_mask = {data.train_mask.shape}')
print(data.train_mask)

print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')

# ============================================================================================
# One of the most helpful utility functions available in PyG is to_networkx.
# It allows you to convert your Data instance into a networkx.Graph to easily visualize it.
# We can use matplotlib to plot the graph with colors corresponding to the label of every node.
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12, 12))
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=0),
                 with_labels=True,
                 node_size=800,
                 node_color=data.y,
                 cmap="hsv",
                 vmin=-2,
                 vmax=3,
                 width=0.8,
                 edge_color="grey",
                 font_size=14
                 )
# plt.show()


# a numpy array with integers instead of floats
X = data.x.numpy().astype(int)
print(f'X = {X.shape}')
print(X)

# learnable weight matrix
W = np.identity(X.shape[0], dtype=int)

print(f'W = {W.shape}')
print(W)

print(f'A = {A.shape}')
print(A)

# A_tilde = A + I
A_tilde = A + np.identity(A.shape[0], dtype=int)

print(f'\nA_tilde = {A_tilde.shape}')
print(A_tilde)


H = A_tilde.T @ X @ W.T

print(f'H = A_tilde.T @ X @ W.T {H.shape}')
print(H)

D = np.zeros(A.shape, dtype=int)
np.fill_diagonal(D, A.sum(axis=0))

print(f'D = {D.shape}')
print(D)

D_tilde = np.zeros(D.shape, dtype=int)
np.fill_diagonal(D_tilde, A_tilde.sum(axis=0))

print(f'\nD_tilde = {D_tilde.shape}')
print(D_tilde)

D_inv = np.linalg.inv(D_tilde)
print(f'D_inv = {D_inv.shape}')
print(D_inv)

H = D_inv @ A_tilde.T @ X @ W.T
print(f'\nH = D_inv @ A.T @ X @ W.T {H.shape}')
print(H)