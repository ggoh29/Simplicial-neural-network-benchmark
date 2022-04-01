import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from models.nn_utils import unpack_feature_dct_to_L_X_B, normalise
from models.GNN.model import GATLayer
from constants import DEVICE
import functools


class SANLayer(nn.Module):

    def __init__(self, input_size, output_size, bias = True):
        super().__init__()
        self.l_d_layer = GATLayer(input_size, output_size, bias)
        self.l_u_layer = GATLayer(input_size, output_size, bias)
        self.p_layer = nn.Linear(input_size, output_size, bias = bias)

    def forward(self, features, l_u, l_d, p):
        h_p = self.p_layer(features)
        h_p = torch.sparse.mm(p, h_p)

        h_u, h_d = torch.zeros(h_p.shape), torch.zeros(h_p.shape)
        if l_u is not None:
            h_u = self.l_d_layer(features, l_u.coalesce().indices())
        if l_d is not None:
            h_d = self.l_d_layer(features, l_d.coalesce().indices())

        return h_u + h_d + h_p



class SuperpixelSAN(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size):
        super().__init__()
        # 10k = 30, 50k = 80
        f_size = 30
        k_heads = 2
        self.layer0_1 = torch.nn.ModuleList([SANLayer(num_node_feats, f_size // k_heads) for _ in range(k_heads)])
        self.layer0_2 = torch.nn.ModuleList([SANLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.layer0_3 = torch.nn.ModuleList([SANLayer(f_size, f_size // k_heads) for _ in range(k_heads)])

        self.layer0_4 = nn.Linear(3 * f_size, output_size)

        self.layer1_1 = torch.nn.ModuleList([SANLayer(num_edge_feats, f_size // k_heads) for _ in range(k_heads)])
        self.layer1_2 = torch.nn.ModuleList([SANLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.layer1_3 = torch.nn.ModuleList([SANLayer(f_size, f_size // k_heads) for _ in range(k_heads)])

        self.layer1_4 = nn.Linear(3 * f_size, output_size)

        self.layer2_1 = torch.nn.ModuleList([SANLayer(num_triangle_feats, f_size // k_heads) for _ in range(k_heads)])
        self.layer2_2 = torch.nn.ModuleList([SANLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.layer2_3 = torch.nn.ModuleList([SANLayer(f_size, f_size // k_heads) for _ in range(k_heads)])

        self.layer2_4 = nn.Linear(3 * f_size, output_size)

        self.combined_layer = nn.Linear(3 * output_size, output_size)

    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        X0, X1, X2 = X
        L0, L1_u, L1_d, L2 = L
        batch0, batch1, batch2 = batch
        l1 = [L1_u, L1_d]

        x0_1 = F.relu(torch.cat([sat(X0, L0) for sat in self.layer0_1], dim=1))
        x0_2 = F.relu(torch.cat([sat(x0_1, L0) for sat in self.layer0_2], dim=1))
        x0_3 = F.relu(torch.cat([sat(x0_2, L0) for sat in self.layer0_3], dim=1))
        x0_4 = self.layer0_4(torch.cat([x0_1, x0_2, x0_3], dim = 1))
        x0 = global_mean_pool(x0_4, batch0)

        x1_1 = F.relu(torch.cat([sat(X1, L1) for L1, sat in zip(l1, self.layer1_1)], dim=1))
        x1_2 = F.relu(torch.cat([sat(x1_1, L1) for L1, sat in zip(l1, self.layer1_2)], dim=1))
        x1_3 = F.relu(torch.cat([sat(x1_2, L1) for L1, sat in zip(l1, self.layer1_3)], dim=1))
        x1_4 = self.layer1_4(torch.cat([x1_1, x1_2, x1_3], dim=1))
        x1 = global_mean_pool(x1_4, batch1)

        x2_1 = F.relu(torch.cat([sat(X2, L2) for sat in self.layer2_1], dim=1))
        x2_2 = F.relu(torch.cat([sat(x2_1, L2) for sat in self.layer2_2], dim=1))
        x2_3 = F.relu(torch.cat([sat(x2_2, L2) for sat in self.layer2_3], dim=1))
        x2_4 = self.layer2_4(torch.cat([x2_1, x2_2, x2_3], dim=1))
        x2 = global_mean_pool(x2_4, batch2)

        x = torch.cat([x0, x1, x2], dim=1)

        return F.softmax(self.combined_layer(x), dim=1)


class FlowSAN(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, f = F.relu, bias = False):
        super().__init__()
        f_size = 32
        self.f = f

        self.layer1 = SANLayer(num_edge_feats, f_size, bias)
        self.layer2 = SANLayer(f_size, f_size, bias)
        self.layer3 = SANLayer(f_size, f_size, bias)
        self.layer4 = SANLayer(f_size, output_size, bias)


    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        _, X1, _ = X
        _, L1_u, L1_d, _ = L
        _, batch1, _= batch
        L1 = L1_u + L1_d

        # Need to obtain sparse approximation of L1
        L = L1.to_dense()
        j = torch.cumsum(torch.bincount(batch1), dim = 0).tolist()
        i = [0] + j[:-1]
        for i,j in zip(i, j):
            L_single = L[i:j,i:j,]
            L_single = normalise(L_single.to_sparse()).to_dense()
            L[i:j,i:j,] = L_single

        L1 = L.to_sparse()

        X1 = self.f(self.layer1(X1, L1_u, L1_d, L1))
        X1 = self.f(self.layer2(X1, L1_u, L1_d, L1))
        X1 = self.f(self.layer3(X1, L1_u, L1_d, L1))
        X1 = self.f(self.layer4(X1, L1_u, L1_d, L1))
        x = global_mean_pool(X1, batch1)

        return F.softmax(x, dim=1)