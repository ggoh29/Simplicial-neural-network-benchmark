import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import functools


class SATLayer(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.a_1 = nn.Linear(output_size, 1, bias=bias)
        self.a_2 = nn.Linear(output_size, 1, bias=bias)
        self.layer = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, features, adj):
        """
        features : n * m dense matrix of feature vectors
        adj : n * n  sparse signed orientation matrix
        output : n * k dense matrix of new feature vectors
        """
        features = self.layer(features)
        indices = adj.coalesce().indices()
        values = adj.coalesce().values()

        a_1 = self.a_1(features.abs())
        a_2 = self.a_2(features.abs())

        v = (a_1 + a_2.T)[indices[0, :], indices[1, :]]
        e = torch.sparse_coo_tensor(indices, v)
        attention = torch.sparse.softmax(e, dim=1)
        a_v = torch.mul(attention.coalesce().values(), values)
        attention = torch.sparse_coo_tensor(indices, a_v)

        output = torch.sparse.mm(attention, features)

        return output


class SuperpixelSAT(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True):
        super().__init__()
        # 10k = 30, 50k = 80
        f_size = 30
        k_heads = 2
        self.layer0_1 = torch.nn.ModuleList([SATLayer(num_node_feats, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer0_2 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer0_3 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])

        self.layer0_4 = nn.Linear(3 * f_size, output_size)

        self.layer1_1 = torch.nn.ModuleList([SATLayer(num_edge_feats, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer1_2 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer1_3 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])

        self.layer1_4 = nn.Linear(3 * f_size, output_size)

        self.layer2_1 = torch.nn.ModuleList(
            [SATLayer(num_triangle_feats, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer2_2 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer2_3 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])

        self.layer2_4 = nn.Linear(3 * f_size, output_size)

        self.combined_layer = nn.Linear(3 * output_size, output_size)

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, _, L2 = simplicialComplex.unpack_laplacians()
        batch0, batch1, batch2 = simplicialComplex.unpack_batch()
        L1 = simplicialComplex.unpack_up_down()

        x0_1 = F.relu(torch.cat([sat(X0, L0) for sat in self.layer0_1], dim=1))
        x0_2 = F.relu(torch.cat([sat(x0_1, L0) for sat in self.layer0_2], dim=1))
        x0_3 = F.relu(torch.cat([sat(x0_2, L0) for sat in self.layer0_3], dim=1))
        x0_4 = self.layer0_4(torch.cat([x0_1, x0_2, x0_3], dim=1))
        x0 = global_mean_pool(x0_4, batch0)

        x1_1 = F.relu(torch.cat([sat(X1, L) for L, sat in zip(L1, self.layer1_1)], dim=1))
        x1_2 = F.relu(torch.cat([sat(x1_1, L) for L, sat in zip(L1, self.layer1_2)], dim=1))
        x1_3 = F.relu(torch.cat([sat(x1_2, L) for L, sat in zip(L1, self.layer1_3)], dim=1))
        x1_4 = self.layer1_4(torch.cat([x1_1, x1_2, x1_3], dim=1))
        x1 = global_mean_pool(x1_4, batch1)

        x2_1 = F.relu(torch.cat([sat(X2, L2) for sat in self.layer2_1], dim=1))
        x2_2 = F.relu(torch.cat([sat(x2_1, L2) for sat in self.layer2_2], dim=1))
        x2_3 = F.relu(torch.cat([sat(x2_2, L2) for sat in self.layer2_3], dim=1))
        x2_4 = self.layer2_4(torch.cat([x2_1, x2_2, x2_3], dim=1))
        x2 = global_mean_pool(x2_4, batch2)

        x = torch.cat([x0, x1, x2], dim=1)

        return F.softmax(self.combined_layer(x), dim=1)


class FlowSAT(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, f=F.relu, bias=False):
        super().__init__()
        f_size = 32
        k_heads = 2

        self.f = f

        self.layer1 = torch.nn.ModuleList([SATLayer(num_edge_feats, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer2 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        # self.layer3 = torch.nn.ModuleList([SATLayer(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer4 = torch.nn.ModuleList([SATLayer(f_size, output_size, bias) for _ in range(k_heads)])

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, _, L2 = simplicialComplex.unpack_laplacians()
        batch1 = simplicialComplex.unpack_batch()[0]
        L1 = simplicialComplex.unpack_up_down()

        X1 = self.f(torch.cat([sat(X1, L) for L, sat in zip(L1, self.layer1)], dim=1))
        X1 = self.f(torch.cat([sat(X1, L) for L, sat in zip(L1, self.layer2)], dim=1))
        # X1 = self.f(torch.cat([sat(X1, L) for L, sat in zip(L1, self.layer3,)], dim=1))
        X1 = self.f(functools.reduce(lambda a, b: a + b, [sat(X1, L) for L, sat in zip(L1, self.layer4)]))
        x = global_mean_pool(X1, batch1)

        return F.softmax(x, dim=1)


class TestSAT(nn.Module):

    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=False, f=F.relu):
        super().__init__()
        k_heads = 2
        self.layer1 = torch.nn.ModuleList([SATLayer(num_node_feats, output_size, bias) for _ in range(k_heads)])
        self.layer2 = torch.nn.ModuleList([SATLayer(num_edge_feats, output_size, bias) for _ in range(k_heads)])
        self.layer3 = torch.nn.ModuleList([SATLayer(num_triangle_feats, output_size, bias) for _ in range(k_heads)])
        self.f = f

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, _, L2 = simplicialComplex.unpack_laplacians()
        L1 = simplicialComplex.unpack_up_down()

        X0 = self.f(functools.reduce(lambda a, b: a + b, [sat(X0, L0) for sat in self.layer1]))
        X1 = self.f(functools.reduce(lambda a, b: a + b, [sat(X1, L) for L, sat in zip(L1, self.layer2)]))
        X2 = self.f(functools.reduce(lambda a, b: a + b, [sat(X2, L2) for sat in self.layer3]))

        return X0, X1, X2
