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

# =============================================================
D_inv12 = np.linalg.inv(D_tilde)
np.fill_diagonal(D_inv12, 1 / (D_tilde.diagonal() ** 0.5))

# New H
H = D_inv12 @ A_tilde.T @ D_inv12 @ X @ W.T
print(f'\nH = D_inv12 @ A.T @ D_inv12 @ X @ W.T {H.shape}')
print(H)


# =============================================================
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index)
        embedding = torch.relu(h)
        z = self.out(embedding)
        return h, embedding, z


model = GNN()
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


# Data for animations
embeddings = []
losses = []
accuracies = []
outputs = []

# Training loop
for epoch in range(500):
    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    h, embedding, z = model(data.x, data.edge_index)
    # print(embedding.detach().shape)

    # Calculate loss function
    loss = criterion(z, data.y)

    # Calculate accuracy
    acc = accuracy(z.argmax(dim=1), data.y)

    # Compute gradients
    loss.backward()

    # Tune parameters
    optimizer.step()

    # Store data for animations
    embeddings.append(embedding)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(z.argmax(dim=1))

    # Print metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc * 100:.2f}%')


# Epoch   0 | Loss: 1.57 | Acc: 8.82%
# Epoch  10 | Loss: 1.53 | Acc: 23.53%
# Epoch  20 | Loss: 1.47 | Acc: 38.24%
# Epoch  30 | Loss: 1.41 | Acc: 38.24%
# Epoch  40 | Loss: 1.34 | Acc: 38.24%
# Epoch  50 | Loss: 1.27 | Acc: 52.94%
# Epoch  60 | Loss: 1.20 | Acc: 67.65%
# Epoch  70 | Loss: 1.14 | Acc: 73.53%
# Epoch  80 | Loss: 1.08 | Acc: 70.59%
# Epoch  90 | Loss: 1.03 | Acc: 70.59%
# Epoch 100 | Loss: 0.99 | Acc: 70.59%
# Epoch 110 | Loss: 0.95 | Acc: 70.59%
# Epoch 120 | Loss: 0.91 | Acc: 70.59%
# Epoch 130 | Loss: 0.87 | Acc: 70.59%
# Epoch 140 | Loss: 0.84 | Acc: 73.53%
# Epoch 150 | Loss: 0.80 | Acc: 73.53%
# Epoch 160 | Loss: 0.76 | Acc: 73.53%
# Epoch 170 | Loss: 0.73 | Acc: 76.47%
# Epoch 180 | Loss: 0.69 | Acc: 82.35%
# Epoch 190 | Loss: 0.65 | Acc: 85.29%
# Epoch 200 | Loss: 0.61 | Acc: 85.29%
# Epoch 210 | Loss: 0.58 | Acc: 85.29%
# Epoch 220 | Loss: 0.54 | Acc: 85.29%
# Epoch 230 | Loss: 0.50 | Acc: 88.24%
# Epoch 240 | Loss: 0.47 | Acc: 97.06%
# Epoch 250 | Loss: 0.44 | Acc: 97.06%
# Epoch 260 | Loss: 0.41 | Acc: 97.06%
# Epoch 270 | Loss: 0.38 | Acc: 97.06%
# Epoch 280 | Loss: 0.35 | Acc: 97.06%
# Epoch 290 | Loss: 0.32 | Acc: 97.06%
# Epoch 300 | Loss: 0.30 | Acc: 97.06%
# Epoch 310 | Loss: 0.28 | Acc: 100.00%
# Epoch 320 | Loss: 0.26 | Acc: 100.00%
# Epoch 330 | Loss: 0.24 | Acc: 100.00%
# Epoch 340 | Loss: 0.22 | Acc: 100.00%
# Epoch 350 | Loss: 0.21 | Acc: 100.00%
# Epoch 360 | Loss: 0.20 | Acc: 100.00%
# Epoch 370 | Loss: 0.18 | Acc: 100.00%
# Epoch 380 | Loss: 0.17 | Acc: 100.00%
# Epoch 390 | Loss: 0.16 | Acc: 100.00%
# Epoch 400 | Loss: 0.15 | Acc: 100.00%
# Epoch 410 | Loss: 0.14 | Acc: 100.00%
# Epoch 420 | Loss: 0.14 | Acc: 100.00%
# Epoch 430 | Loss: 0.13 | Acc: 100.00%
# Epoch 440 | Loss: 0.12 | Acc: 100.00%
# Epoch 450 | Loss: 0.12 | Acc: 100.00%
# Epoch 460 | Loss: 0.11 | Acc: 100.00%
# Epoch 470 | Loss: 0.10 | Acc: 100.00%
# Epoch 480 | Loss: 0.10 | Acc: 100.00%
# Epoch 490 | Loss: 0.09 | Acc: 100.00%
#
# Process finished with exit code 0
