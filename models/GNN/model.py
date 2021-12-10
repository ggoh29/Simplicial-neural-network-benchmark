import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from models.nn_utils import chebyshev, unpack_feature_dct_to_L_X_B
from constants import DEVICE

class GCNLayer(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias = True):
        super().__init__()
        self.conv = nn.Linear(feature_size, output_size, bias = enable_bias)

    def forward(self, L, X):
        X = torch.sparse.mm(L, X)
        return self.conv(X)

class GCN1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 96
        self.conv1 = GCNConv(input_size, f_size)
        self.conv2 = GCNConv(f_size, f_size)
        self.conv3 = GCNConv(f_size, 30)
        self.layer_final = nn.Linear(30, output_size)

    def forward(self, feature_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(feature_dct)

        adjacency = L[0].coalesce().indices()
        weights = torch.abs(L[0].coalesce().values())
        x = X[0]

        x = F.relu(self.conv1(x, adjacency, weights))
        x = F.relu(self.conv2(x, adjacency, weights))
        x = F.relu(self.conv3(x, adjacency, weights))

        # x = torch.cat([x1, x2 ,x3], dim = 1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer_final(x), dim = 1)



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
        weights = L[0].coalesce().values()
        features = X[0]

        x1 = F.relu(self.conv1(features, adjacency, weights))
        x2 = F.relu(self.conv2(x1, adjacency, weights))
        x3 = F.relu(self.conv3(x2, adjacency, weights))

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
        weights = L[0].coalesce().values()
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
        # self.a = nn.Linear(2 * input_size, 1)
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, features, adj):

        n = features.size()[0]
        features = self.layer(features)

        a_1 = self.a_1(features)
        a_2 = self.a_2(features)

        e = (a_1 + a_2.T)

        zero_vec = -1e16 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(nn.LeakyReLU()(attention), dim = 1).unsqueeze(2)

        attr = torch.stack([features for _ in range(n)], dim = 0)
        output = attr * attention
        output = output.sum(dim = 1)
        return output


class GAT1(nn.Module):
    def __init__(self, input_size, output_size, k_heads = 2):
        super().__init__()

        f_size = 64
        assert f_size % k_heads == 0, f"k_heads needs to be a factor of feature size which is currently {f_size}."
        self.gat1 = torch.nn.ModuleList([GATLayer(input_size, f_size//k_heads) for _ in range(k_heads)])
        self.gat2 = torch.nn.ModuleList([GATLayer(f_size, f_size//k_heads) for _ in range(k_heads)])
        self.gat3 = torch.nn.ModuleList([GATLayer(f_size, f_size//k_heads) for _ in range(k_heads)])
        self.layer = nn.Linear(3 * f_size, output_size)

    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        adjacency = L[0].to_dense()
        x = X[0]

        # edgelist = L[0].coalesce().indices()
        # n = edgelist.shape[1]
        # values = torch.ones(n).to(DEVICE)
        # adjacency = torch.sparse_coo_tensor(edgelist, values).to_dense()

        x1 = F.relu(torch.cat([gat(x, adjacency) for gat in self.gat1], dim = 1))
        x2 = F.relu(torch.cat([gat(x1, adjacency) for gat in self.gat2], dim = 1))
        x3 = F.relu(torch.cat([gat(x2, adjacency) for gat in self.gat3], dim = 1))

        x = torch.cat([x1, x2, x3], dim=1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer(x), dim=1)


