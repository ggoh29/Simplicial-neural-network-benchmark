import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import functools


class SATLayer_orientated(nn.Module):

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


class SATLayer_regular(nn.Module):

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

        a_1 = self.a_1(features)
        a_2 = self.a_2(features)

        v = (a_1 + a_2.T)[indices[0, :], indices[1, :]]
        e = torch.sparse_coo_tensor(indices, v)
        attention = torch.sparse.softmax(e, dim=1)

        output = torch.sparse.mm(attention, features)

        return output


class SuperpixelSAT(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True):
        super().__init__()
        # 10k = 30, 50k = 80
        f_size = 30
        k_heads = 2
        self.layer0_1 = torch.nn.ModuleList(
            [SATLayer_regular(num_node_feats, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer0_2 = torch.nn.ModuleList([SATLayer_regular(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer0_3 = torch.nn.ModuleList([SATLayer_regular(f_size, f_size // k_heads, bias) for _ in range(k_heads)])

        self.layer0_4 = nn.Linear(3 * f_size, output_size)

        self.layer1_1 = torch.nn.ModuleList(
            [SATLayer_regular(num_edge_feats, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer1_2 = torch.nn.ModuleList([SATLayer_regular(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer1_3 = torch.nn.ModuleList([SATLayer_regular(f_size, f_size // k_heads, bias) for _ in range(k_heads)])

        self.layer1_4 = nn.Linear(3 * f_size, output_size)

        self.layer2_1 = torch.nn.ModuleList(
            [SATLayer_regular(num_triangle_feats, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer2_2 = torch.nn.ModuleList([SATLayer_regular(f_size, f_size // k_heads, bias) for _ in range(k_heads)])
        self.layer2_3 = torch.nn.ModuleList([SATLayer_regular(f_size, f_size // k_heads, bias) for _ in range(k_heads)])

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


class PRELU(nn.PReLU):

    def forward(self, input):
        return F.prelu(input, self.weight)


class PlanetoidSAT(nn.Module):

    def __init__(self, num_node_feats, output_size, bias=True):
        super().__init__()
        k_heads = 2
        self.layer_n = torch.nn.ModuleList([SATLayer_regular(num_node_feats, output_size, bias) for _ in range(k_heads)])
        self.layer_e = torch.nn.ModuleList([SATLayer_regular(num_node_feats, output_size, bias) for _ in range(k_heads)])
        self.layer_t = torch.nn.ModuleList([SATLayer_regular(num_node_feats, output_size, bias) for _ in range(k_heads)])
        self.f = PRELU()

        self.tri_layer = nn.Linear(output_size, output_size)

    def forward(self, simplicialComplex, B1, B2):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, _, L2 = simplicialComplex.unpack_laplacians()
        L1 = simplicialComplex.unpack_up_down()

        X0[X0 != 0] = 1

        X1_in, X1_out = X0[X1[:, 0]], X0[X1[:, 1]]
        X1 = torch.logical_and(X1_in, X1_out).float()

        X2_i, X2_j, X2_k = X0[X2[:, 0]], X0[X2[:, 1]], X0[X2[:, 2]]
        X2 = torch.logical_and(X2_i, torch.logical_and(X2_j, X2_k)).float()

        X0 = self.f(functools.reduce(lambda a, b: a + b, [sat(X0, L0) for sat in self.layer_n]))
        X1 = self.f(functools.reduce(lambda a, b: a + b, [sat(X1, L) for L, sat in zip(L1, self.layer_e)]))
        X2 = self.f(functools.reduce(lambda a, b: a + b, [sat(X2, L2) for sat in self.layer_t]))

        X0 = (X0 + torch.sparse.mm(B1, X1) + torch.sparse.mm(B1, self.tri_layer(torch.sparse.mm(B2, X2)))) / 3
        return X0


class FlowSAT(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, f=F.relu, bias=False):
        super().__init__()
        f_size = 32
        k_heads = 2

        self.f = f

        self.layer1 = torch.nn.ModuleList(
            [SATLayer_orientated(num_edge_feats, f_size // 2, bias) for _ in range(k_heads)])
        self.layer2 = torch.nn.ModuleList([SATLayer_orientated(f_size, f_size // 2, bias) for _ in range(k_heads)])
        self.layer3 = torch.nn.ModuleList([SATLayer_orientated(f_size, f_size // 2, bias) for _ in range(k_heads)])
        self.layer4 = torch.nn.ModuleList([SATLayer_orientated(f_size, f_size, bias) for _ in range(k_heads)])

        self.mlp1 = nn.Linear(f_size, f_size)
        self.mlp2 = nn.Linear(f_size, output_size)

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, _, L2 = simplicialComplex.unpack_laplacians()
        batch1 = simplicialComplex.unpack_batch()[1]
        L1 = simplicialComplex.unpack_up_down()

        X1 = self.f(torch.cat([sat(X1, L) for L, sat in zip(L1, self.layer1)], dim=1))
        X1 = self.f(torch.cat([sat(X1, L) for L, sat in zip(L1, self.layer2)], dim=1))
        X1 = self.f(torch.cat([sat(X1, L) for L, sat in zip(L1, self.layer3)], dim=1))
        X1 = self.f(functools.reduce(lambda a, b: a + b, [sat(X1, L) for L, sat in zip(L1, self.layer4)]))

        X1 = global_mean_pool(X1.abs(), batch1)
        X1 = F.relu(self.mlp1(X1))

        return torch.softmax(self.mlp2(X1), dim=1)


class TestSAT(nn.Module):

    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=False, f=F.relu):
        super().__init__()
        k_heads = 2
        self.layer1 = torch.nn.ModuleList(
            [SATLayer_orientated(num_node_feats, output_size, bias) for _ in range(k_heads)])
        self.layer2 = torch.nn.ModuleList(
            [SATLayer_orientated(num_edge_feats, output_size, bias) for _ in range(k_heads)])
        self.layer3 = torch.nn.ModuleList(
            [SATLayer_orientated(num_triangle_feats, output_size, bias) for _ in range(k_heads)])
        self.f = f

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, _, L2 = simplicialComplex.unpack_laplacians()
        L1 = simplicialComplex.unpack_up_down()

        X0 = self.f(functools.reduce(lambda a, b: a + b, [sat(X0, L0) for sat in self.layer1]))
        X1 = self.f(functools.reduce(lambda a, b: a + b, [sat(X1, L) for L, sat in zip(L1, self.layer2)]))
        X2 = self.f(functools.reduce(lambda a, b: a + b, [sat(X2, L2) for sat in self.layer3]))

        return X0, X1, X2
