import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from models.nn_utils import chebyshev, unpack_feature_dct_to_L_X_B, convert_indices_and_values_to_sparse


class GCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.conv1 = GCNConv(input_size, f_size)
        self.conv2 = GCNConv(f_size, f_size)
        self.conv3 = GCNConv(f_size, f_size)
        self.layer1 = nn.Linear(f_size, output_size)
        self.layer2 = nn.Linear(f_size, output_size)
        self.layer3 = nn.Linear(f_size, output_size)
        self.layer_final = nn.Linear(3 * output_size, output_size)

    def forward(self, feature_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(feature_dct)

        adjacency = L[0].coalesce().indices()
        # weights = torch.abs(L[0].coalesce().values())
        # features = chebyshev(L[0], X[0])
        features = X[0]

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x1 = self.layer1(global_mean_pool(x1, batch[0]))
        x2 = self.layer2(global_mean_pool(x2, batch[0]))
        x3 = self.layer3(global_mean_pool(x3, batch[0]))

        x = torch.cat([x1, x2, x3], dim = 1)

        return F.softmax(self.layer_final(x), dim = 1)



class GAT(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        f_size = 64
        self.conv1 = GATConv(input_size, f_size)
        self.conv2 = GATConv(f_size, f_size)
        self.conv3 = GATConv(f_size, f_size)
        self.layer = nn.Linear(f_size, output_size)

    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        adjacency = L[0].coalesce().indices()
        # weights = torch.abs(L[0].coalesce().values())
        features = X[0]

        x1 = F.relu(self.conv1(features, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = global_mean_pool((x1 + x2 + x3)/3, batch[0])
        return F.softmax(self.layer(x), dim = 1)