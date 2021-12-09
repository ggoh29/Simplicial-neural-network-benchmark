import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from models.nn_utils import chebyshev, unpack_feature_dct_to_L_X_B, convert_indices_and_values_to_sparse
from constants import DEVICE
import numpy as np


class GCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.conv1 = GCNConv(input_size, f_size)
        self.conv2 = GCNConv(f_size, f_size)
        self.conv3 = GCNConv(f_size, f_size)
        self.layer_final = nn.Linear(3 * f_size, output_size)

    def forward(self, feature_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(feature_dct)

        adjacency = L[0].coalesce().indices()
        # weights = torch.abs(L[0].coalesce().values())
        # features = chebyshev(L[0], X[0])
        features = X[0]

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = torch.cat([x1, x2 ,x3], dim = 1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer_final(x), dim = 1)



class GAT(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.conv1 = GATConv(input_size, f_size)
        self.conv2 = GATConv(f_size, f_size)
        self.conv3 = GATConv(f_size, f_size)
        self.layer = nn.Linear(3 * f_size, output_size)

    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        adjacency = L[0].coalesce().indices()
        # weights = torch.abs(L[0].coalesce().values())
        features = X[0]

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = torch.cat([x1, x2 ,x3], dim = 1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer(x), dim = 1)


class GATLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.a = nn.Linear(2 * output_size, 1)
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, features, edgelist):
        features = self.layer(features)
        n = torch.max(edgelist).item() + 1

        src = features.unsqueeze(0).expand(n, -1, -1)
        tgt = features.unsqueeze(1).expand(-1, n, -1)
        h = torch.cat([src,tgt],dim=2)

        attention = self.a(h).squeeze(2)
        # attention = F.normalize(attention, dim = 1)
        attention = F.softmin(attention, dim = 1)
        print(attention)
        return torch.matmul(attention, features)


class GAT1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 6
        self.gat1 = GATLayer(input_size, f_size)
        self.gat2 = GATLayer(f_size, f_size)
        self.gat3 = GATLayer(f_size, f_size)
        self.layer = nn.Linear(3 * f_size, output_size)

    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        adjacency = L[0].coalesce().indices()
        # weights = torch.abs(L[0].coalesce().values())
        features = X[0]

        x1 = F.relu(self.gat1(features, adjacency))
        x2 = F.relu(self.gat2(x1, adjacency))
        x3 = F.relu(self.gat3(x2, adjacency))

        x = torch.cat([x1, x2, x3], dim=1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer(x), dim=1)