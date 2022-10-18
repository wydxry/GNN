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
plt.show()
