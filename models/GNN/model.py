import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class SuperpixelGCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 10k = 64, 50k = 148
        f_size = 64
        self.conv1 = GCNConv(input_size, f_size, add_self_loops=False)
        self.conv2 = GCNConv(f_size, f_size, add_self_loops=False)
        self.conv3 = GCNConv(f_size, f_size, add_self_loops=False)
        self.layer_final = nn.Linear(3 * f_size, output_size)

    def forward(self, simplicialComplex):
        X0, _, _ = simplicialComplex.unpack_features()
        L0, _, _ = simplicialComplex.unpack_laplacians()
        batch = simplicialComplex.unpack_batch()

        adjacency = L0.coalesce().indices()

        x1 = F.relu(self.conv1(X0, adjacency))
        x2 = F.relu(self.conv2(x1, adjacency))
        x3 = F.relu(self.conv3(x2, adjacency))

        x = torch.cat([x1, x2, x3], dim=1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer_final(x), dim=1)


class PlanetoidGCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.act = nn.PReLU()
        self.conv1 = GCNConv(input_size, output_size, add_self_loops=False)

    def forward(self, simplicialComplex):
        X0, _, _ = simplicialComplex.unpack_features()
        L0, _, _ = simplicialComplex.unpack_laplacians()

        x = self.act(self.conv1(X0, L0.coalesce().indices()))

        return x


class GATLayer(nn.Module):

    def __init__(self, input_size, output_size, bias = True):
        super().__init__()
        self.a_1 = nn.Linear(output_size, 1, bias = bias)
        self.a_2 = nn.Linear(output_size, 1, bias = bias)
        self.layer = nn.Linear(input_size, output_size, bias = bias)

    def forward(self, features, indices):
        features = self.layer(features)

        a_1 = self.a_1(features)
        a_2 = self.a_2(features)

        v = (a_1 + a_2.T)[indices[0, :], indices[1, :]]
        v = nn.LeakyReLU()(v)
        e = torch.sparse_coo_tensor(indices, v)
        attention = torch.sparse.softmax(e, dim=1)

        output = torch.sparse.mm(attention, features)

        return output


class SuperpixelGAT(nn.Module):
    def __init__(self, input_size, output_size, k_heads=2):
        super().__init__()
        # 10k = 60, 50k = 148
        f_size = 60
        assert f_size % k_heads == 0, f"k_heads needs to be a factor of feature size which is currently {f_size}."
        self.gat1 = torch.nn.ModuleList([GATLayer(input_size, f_size // k_heads) for _ in range(k_heads)])
        self.gat2 = torch.nn.ModuleList([GATLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.gat3 = torch.nn.ModuleList([GATLayer(f_size, f_size // k_heads) for _ in range(k_heads)])
        self.layer = nn.Linear(3 * f_size, output_size)

    def forward(self, simplicialComplex):
        X0, _, _ = simplicialComplex.unpack_features()
        L0, _, _ = simplicialComplex.unpack_laplacians()
        batch = simplicialComplex.unpack_batch()
        adjacency = L0.coalesce().indices()

        x1 = F.relu(torch.cat([gat(X0, adjacency) for gat in self.gat1], dim=1))
        x2 = F.relu(torch.cat([gat(x1, adjacency) for gat in self.gat2], dim=1))
        x3 = F.relu(torch.cat([gat(x2, adjacency) for gat in self.gat3], dim=1))

        x = torch.cat([x1, x2, x3], dim=1)
        x = global_mean_pool(x, batch[0])

        return F.softmax(self.layer(x), dim=1)


class PlanetoidGAT(nn.Module):

    def __init__(self, input_size, output_size, k_heads=2):
        super().__init__()

        f_size = output_size // 2
        assert f_size % k_heads == 0, f"k_heads needs to be a factor of feature size which is currently {f_size}."
        self.gat1 = torch.nn.ModuleList([GATLayer(input_size, output_size // k_heads) for _ in range(k_heads)])

    def forward(self, simplicialComplex):
        X0, _, _ = simplicialComplex.unpack_features()
        L0, _, _ = simplicialComplex.unpack_laplacians()
        adjacency = L0.coalesce().indices()

        x = F.relu(torch.cat([gat(X0, adjacency) for gat in self.gat1], dim=1))

        return x
