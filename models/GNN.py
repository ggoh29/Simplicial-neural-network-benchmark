import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_scatter import scatter_max


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


class GCNConv1(nn.Module):
    def __init__(self, num_node_feats, output_node_dim, bias):
        super(GCNConv1, self).__init__()
        self.num_node_feats = num_node_feats
        self.output_node_dim = output_node_dim
        self.bias = bias

        self.n2n_weights = nn.Linear(
            self.num_node_feats,
            self.output_node_dim,
            bias=self.bias,
        )

    def forward(self, X0, L0):
        n2n = self.n2n_weights(X0)
        n2n = torch.sparse.mm(L0, n2n)
        return F.relu(n2n)


class GCN2(nn.Module):
    def __init__(self, input_size, output_size, batch_size):
        super().__init__()
        self.conv1 = GCNConv1(input_size, 64, True)
        self.conv2 = GCNConv1(64, 64, True)
        self.conv3 = GCNConv1(64, 64, True)
        self.layer = nn.Linear(64, output_size, bias = True)

    def forward(self, X, L, batch):
        L = L[0]
        X = X[0].type(torch.float32)

        X1 = self.conv1(X, L)
        X2 = self.conv2(X1, L)
        X3 = self.conv3(X2, L)

        x = global_mean_pool((X1 + X2 + X3)/3, batch[0])
        return F.softmax(self.layer(x), dim = 1)


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
        features = X[0]

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = global_mean_pool((x1 + x2 + x3)/3, batch[0])
        return F.softmax(self.layer(x), dim = 1)