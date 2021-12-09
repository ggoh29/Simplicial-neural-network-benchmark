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
        self.a_1 = nn.Linear(output_size, 1)
        self.a_2 = nn.Linear(output_size, 1)
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, features, adj):
        features = self.layer(features).detach()

        a_1 = self.a_1(features)
        a_2 = self.a_2(features)
        e = (a_1 + a_2.T)

        zero_vec = -1e16 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim = 1)
        return torch.matmul(attention, features)


class GAT1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.gat1 = GATLayer(input_size, f_size)
        self.gat2 = GATLayer(f_size, f_size)
        self.gat3 = GATLayer(f_size, f_size)
        self.layer = nn.Linear(3 * f_size, output_size)

    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        edgelist = L[0].coalesce().indices()
        # weights = torch.abs(L[0].coalesce().values())
        features = X[0]

        n = edgelist.shape[1]
        adjacency = torch.sparse_coo_tensor(edgelist, torch.ones(n)).to_dense()

        x1 = F.relu(self.gat1(features, adjacency))
        x2 = F.relu(self.gat2(x1, adjacency))
        x3 = F.relu(self.gat3(x2, adjacency))

        x = torch.cat([x1, x2, x3], dim=1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer(x), dim=1)
