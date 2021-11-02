import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_scatter import scatter_max


class GCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 16
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


class GCN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 48)
        self.conv4 = GCNConv(48, 64)
        self.conv5 = GCNConv(64, 96)
        self.conv6 = GCNConv(96, 128)
        self.linear1 = torch.nn.Linear(128,64)
        self.linear2 = torch.nn.Linear(64,10)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x, _ = scatter_max(x, batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.softmax(self.linear2(x), dim = 1)
        return x


class GCN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(5, 64, bias = True)
        self.conv2 = nn.Linear(64, 64, bias = True)
        self.conv3 = nn.Linear(64, 64, bias = True)
        self.layer = nn.Linear(64, 10, bias = True)

    def forward(self, X, L, batch):
        L = L[0]
        x = X[0]

        x1 = F.relu(self.conv1(torch.sparse.mm(L, x)))
        x2 = F.relu(self.conv2(torch.sparse.mm(L, x1)))
        x3 = F.relu(self.conv3(torch.sparse.mm(L, x2)))

        x = global_mean_pool((x1 + x2 + x3)/3, batch[0])
        return F.softmax(self.layer(x), dim = 1)