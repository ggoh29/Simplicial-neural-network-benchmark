import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from models.nn_utils import chebyshev



class GCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.conv1 = GCNConv(input_size, f_size)
        self.conv2 = GCNConv(f_size, f_size)
        self.conv3 = GCNConv(f_size, f_size)
        self.layer = nn.Linear(f_size, output_size)

    def forward(self, X, L, batch):
        adjacency = L[0].coalesce().indices()
        # weights = L[0].coalesce().values()
        features = X[0]

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = global_mean_pool((x1 + x2 + x3)/3, batch[0])
        return F.softmax(self.layer(x), dim = 1)

    def normalise(self, L):
        return L


class GCN2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.conv1 = GCNConv(3 * input_size, f_size)
        self.conv2 = GCNConv(f_size, f_size)
        self.conv3 = GCNConv(f_size, f_size)
        self.layer = nn.Linear(f_size, output_size)

    def forward(self, X, L, batch):
        adjacency = L[0].coalesce().indices()
        # weights = L[0].coalesce().values()
        features = chebyshev(L[0], X[0])

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = global_mean_pool((x1 + x2 + x3)/3, batch[0])
        return F.softmax(self.layer(x), dim = 1)

    def normalise(self, L):
        return L


class GCN3(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.conv1 = GATConv(input_size, f_size)
        self.conv2 = GATConv(f_size, f_size)
        self.conv3 = GATConv(f_size, f_size)
        self.layer = nn.Linear(f_size, output_size)

    def forward(self, X, L, batch):
        adjacency = L[0].coalesce().indices()
        # weights = L[0].coalesce().values()
        features =  X[0]

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = global_mean_pool((x1 + x2 + x3)/3, batch[0])
        return F.softmax(self.layer(x), dim = 1)

    def normalise(self, L):
        return L