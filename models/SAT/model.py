import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from models.nn_utils import unpack_feature_dct_to_L_X_B
import functools
from constants import DEVICE


class SATLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.a_1 = nn.Linear(output_size, 1)
        self.a_2 = nn.Linear(output_size, 1)
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, features, adj):

        n = features.size()[0]
        features = self.layer(features)

        indices = adj.coalesce().indices()
        values = adj.coalesce().values()

        a_1 = self.a_1(features)
        a_2 = self.a_2(features)

        v = torch.tensor([a_1[i.item()] + a_2[j.item()] for i,j in zip(indices[0], indices[1])], dtype=torch.float, device=DEVICE)
        v = nn.LeakyReLU()(v)
        e1 = torch.sparse_coo_tensor(indices, v)
        attention = torch.sparse.softmax(e1, dim = 1)
        attr = torch.stack([features for _ in range(n)], dim=0)

        attr_sparse = attr[indices[0, :], indices[1, :]]
        attr_sparse = attention.coalesce().values().unsqueeze(dim = 1) * attr_sparse
        attr_sparse = torch.sparse_coo_tensor(indices, attr_sparse)
        output = torch.sparse.sum(attr_sparse, dim = 1).to_dense()

        return output


class SAT(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size):
        super().__init__()

        f_size = 48
        k_heads = 2
        self.layer0_1 = torch.nn.ModuleList([SATLayer(num_node_feats, f_size//k_heads) for _ in range(k_heads)])
        self.layer0_2 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.layer0_3 = torch.nn.ModuleList([SATLayer(f_size, output_size) for _ in range(k_heads)])

        self.layer1_1 = torch.nn.ModuleList([SATLayer(num_edge_feats, f_size // k_heads) for _ in range(k_heads)])
        self.layer1_2 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.layer1_3 = torch.nn.ModuleList([SATLayer(f_size, output_size) for _ in range(k_heads)])

        self.layer2_1 = torch.nn.ModuleList([SATLayer(num_triangle_feats, f_size // k_heads) for _ in range(k_heads)])
        self.layer2_2 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.layer2_3 = torch.nn.ModuleList([SATLayer(f_size, output_size) for _ in range(k_heads)])

        self.layer = nn.Linear(3 * output_size, output_size)

    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        X0, X1, _, X2 = X
        L0, L1_u, L1_d, L2 = L
        batch0, batch1, _, batch2 = batch
        # L0, L1_u, L1_d, L2 = L0.to_dense(), L1_u.to_dense(), L1_d.to_dense(), L2.to_dense()
        l1 = [L1_u, L1_d]

        x0_1 = F.relu(torch.cat([sat(X0, L0) for sat in self.layer0_1], dim = 1))
        x0_2 = F.relu(torch.cat([sat(x0_1, L0) for sat in self.layer0_2], dim = 1))
        x0_3 = functools.reduce(lambda a, b: a + b, [sat(x0_2, L0) for sat in self.layer0_3])/2
        x0 = global_mean_pool(x0_3, batch0)

        x1_1 = F.relu(torch.cat([sat(X1, L1) for L1, sat in zip(l1, self.layer1_1)], dim=1))
        x1_2 = F.relu(torch.cat([sat(x1_1, L1) for L1, sat in zip(l1, self.layer1_2)], dim=1))
        x1_3 = functools.reduce(lambda a, b: a + b, [sat(x1_2, L1) for L1, sat in zip(l1, self.layer1_3)])/2
        x1 = global_mean_pool(x1_3, batch1)

        x2_1 = F.relu(torch.cat([sat(X2, L2) for sat in self.layer2_1], dim=1))
        x2_2 = F.relu(torch.cat([sat(x2_1, L2) for sat in self.layer2_2], dim=1))
        x2_3 = functools.reduce(lambda a, b: a + b, [sat(x2_2, L2) for sat in self.layer2_3])/2
        x2 = global_mean_pool(x2_3, batch2)

        x = torch.cat([x0, x1, x2], dim=1)

        return F.softmax(self.layer(x), dim=1)