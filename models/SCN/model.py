import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from models.nn_utils import chebyshev


class SCNLayer(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias=True, k=1):
        super().__init__()
        self.k = k
        self.conv = nn.Linear(k * feature_size, output_size, bias=enable_bias)

    def forward(self, L, x):
        X = chebyshev(L, x, self.k)
        return self.conv(X)


class SuperpixelSCN(nn.Module):
    # This model is based on model described by Stefanie Ebli et al. in Simplicial Neural Networks
    # Github here https://github.com/stefaniaebli/simplicial_neural_networks?utm_source=catalyzex.com
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True):
        super().__init__()

        conv_size = 32

        # Degree 0 convolutions.
        self.C0_1 = SCNLayer(num_node_feats, conv_size, enable_bias=bias)
        self.C0_2 = SCNLayer(conv_size, conv_size, enable_bias=bias)
        self.C0_3 = SCNLayer(conv_size, conv_size, enable_bias=bias)

        # Degree 1 convolutions.
        self.C1_1 = SCNLayer(num_edge_feats, conv_size, enable_bias=bias)
        self.C1_2 = SCNLayer(conv_size, conv_size, enable_bias=bias)
        self.C1_3 = SCNLayer(conv_size, conv_size, enable_bias=bias)

        # Degree 2 convolutions.
        self.C2_1 = SCNLayer(num_triangle_feats, conv_size, enable_bias=bias)
        self.C2_2 = SCNLayer(conv_size, conv_size, enable_bias=bias)
        self.C2_3 = SCNLayer(conv_size, conv_size, enable_bias=bias)

        self.layer0 = nn.Linear(3 * conv_size, output_size)
        self.layer1 = nn.Linear(3 * conv_size, output_size)
        self.layer2 = nn.Linear(3 * conv_size, output_size)

        self.combined_layer = nn.Linear(output_size * 3, output_size)

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, L1, L2 = simplicialComplex.unpack_laplacians()
        batch = simplicialComplex.unpack_batch()

        out0_1 = nn.LeakyReLU()(self.C0_1(L0, X0))
        out0_2 = nn.LeakyReLU()(self.C0_2(L0, out0_1))
        out0_3 = nn.LeakyReLU()(self.C0_3(L0, out0_2))

        out1_1 = nn.LeakyReLU()(self.C1_1(L1, X1))
        out1_2 = nn.LeakyReLU()(self.C1_2(L1, out1_1))
        out1_3 = nn.LeakyReLU()(self.C1_3(L1, out1_2))

        out2_1 = nn.LeakyReLU()(self.C2_1(L2, X2))
        out2_2 = nn.LeakyReLU()(self.C2_2(L2, out2_1))
        out2_3 = nn.LeakyReLU()(self.C2_3(L2, out2_2))

        out0 = self.layer0(torch.cat([out0_1, out0_2, out0_3], dim=1))
        out1 = self.layer1(torch.cat([out1_1, out1_2, out1_3], dim=1))
        out2 = self.layer2(torch.cat([out2_1, out2_2, out2_3], dim=1))

        out0 = global_mean_pool(out0, batch[0])
        out1 = global_mean_pool(out1, batch[1])
        out2 = global_mean_pool(out2, batch[2])

        out = torch.cat([out0, out1, out2], dim=1)
        return F.softmax(self.combined_layer(out), dim=1)

class PRELU(nn.PReLU):

    def forward(self, input):
        return F.prelu(input, self.weight)


class PlanetoidSCN(nn.Module):

    def __init__(self, num_node_feats, output_size, bias=True):
        super().__init__()
        f_size = output_size
        self.layer_n = SCNLayer(num_node_feats, f_size, bias)
        self.layer_e = SCNLayer(num_node_feats, f_size, bias)
        self.layer_t = SCNLayer(num_node_feats, f_size, bias)
        self.f = PRELU()

        self.tri_layer = nn.Linear(output_size, output_size)

    def forward(self, simplicialComplex, B1, B2):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, L1, L2 = simplicialComplex.unpack_laplacians()

        X0[X0 != 0] = 1

        X1_in, X1_out = X0[X1[:, 0]], X0[X1[:, 1]]
        X1 = torch.logical_and(X1_in, X1_out).float()

        X2_i, X2_j, X2_k = X0[X2[:, 0]], X0[X2[:, 1]], X0[X2[:, 2]]
        X2 = torch.logical_and(X2_i, torch.logical_and(X2_j, X2_k)).float()

        X0 = self.f(self.layer_n(L0, X0))
        X1 = self.f(self.layer_e(L1, X1))
        X2 = self.f(self.layer_t(L2, X2))

        X0 = (X0 + torch.sparse.mm(B1, X1) + torch.sparse.mm(B1, self.tri_layer(torch.sparse.mm(B2, X2)))) / 3
        return X0


class FlowSCN(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=False, f=nn.LeakyReLU()):
        super().__init__()

        conv_size = 32

        self.layer1 = SCNLayer(num_edge_feats, conv_size, enable_bias=bias)
        self.layer2 = SCNLayer(conv_size, conv_size, enable_bias=bias)
        self.layer3 = SCNLayer(conv_size, conv_size, enable_bias=bias)
        self.layer4 = SCNLayer(conv_size, conv_size, enable_bias=bias)
        self.mlp1 = nn.Linear(conv_size, conv_size)
        self.mlp2 = nn.Linear(conv_size, output_size)
        self.f = f

    def forward(self, simplicialComplex):
        _, X1, _ = simplicialComplex.unpack_features()
        _, L1, _ = simplicialComplex.unpack_laplacians()
        batch = simplicialComplex.unpack_batch()

        X1 = self.f(self.layer1(L1, X1))
        X1 = self.f(self.layer2(L1, X1))
        X1 = self.f(self.layer3(L1, X1))
        X1 = self.f(self.layer4(L1, X1))

        X1 = global_mean_pool(X1.abs(), batch[1])
        X1 = F.relu(self.mlp1(X1))

        return F.softmax(self.mlp2(X1), dim=1)


class TestSCN(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=False, f=nn.Identity()):
        super().__init__()
        self.layer1 = SCNLayer(num_node_feats, output_size, enable_bias=bias)
        self.layer2 = SCNLayer(num_edge_feats, output_size, enable_bias=bias)
        self.layer3 = SCNLayer(num_triangle_feats, output_size, enable_bias=bias)
        self.f = f

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, L1, L2 = simplicialComplex.unpack_laplacians()

        X0 = self.f(self.layer1(L0, X0))
        X1 = self.f(self.layer2(L1, X1))
        X2 = self.f(self.layer3(L2, X2))

        return X0, X1, X2
