import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class SCConv_Layer(nn.Module):
    # This model is based on model described by Eric Bunch et al. in Simplicial 2-Complex Convolutional Neural Networks
    # Github here https://github.com/AmFamMLTeam/simplicial-2-complex-cnns
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True, f=F.relu):
        super().__init__()
        self.n2n_weights = nn.Linear(num_node_feats, output_size, bias=bias)
        self.n2e_weights = nn.Linear(num_node_feats, output_size, bias=bias)
        self.e2e_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
        self.e2n_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
        self.e2t_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
        self.t2e_weights = nn.Linear(num_triangle_feats, output_size, bias=bias)
        self.t2t_weights = nn.Linear(num_triangle_feats, output_size, bias=bias)
        self.w = f

    def forward(self, X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv):
        # node to node
        n2n = self.n2n_weights(X0)  # Y00 = X0*W00
        n2n = torch.sparse.mm(L0, n2n)  # L0*Y00

        # node to edge
        n2e = self.n2e_weights(X0)  # Y01 = X0*W01
        n2e = torch.sparse.mm(D2B1TD1inv, n2e)  # D2*B1.T*D1^-1*Y01

        # edge to node
        e2n = self.e2n_weights(X1)  # Y10 = X1*W10
        e2n = torch.sparse.mm(D1invB1, e2n)  # D1invB1*Y10

        # edge to edge
        e2e = self.e2e_weights(X1)  # Y11 = X1*W11
        e2e = torch.sparse.mm(L1, e2e)  # L1*Y11

        # edge to triangle
        e2t = self.e2t_weights(X1)  # Y21 = X1*W21
        e2t = torch.sparse.mm(B2TD2inv, e2t)  # B2TD2inv*Y21

        # triangle to triangle
        t2t = self.t2t_weights(X2)  # Y22 = X2*W22
        t2t = torch.sparse.mm(L2, t2t)  # L2*Y22

        # triangle to edge
        t2e = self.t2e_weights(X2)  # Y12 = X2*W12
        t2e = torch.sparse.mm(B2D3, t2e)  # B2D3*Y12

        X0 = (1 / 2.) * self.w(n2n + e2n)
        X1 = (1 / 3.) * self.w(e2e + n2e + t2e)
        X2 = (1 / 2.) * self.w(t2t + e2t)

        return X0, X1, X2


class SuperpixelSCConv(nn.Module):

    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True):
        super().__init__()
        # 10k = 20, 50k = 54
        f_size = 21
        self.layer1 = SCConv_Layer(num_node_feats, num_edge_feats, num_triangle_feats, f_size, bias)
        self.layer2 = SCConv_Layer(f_size, f_size, f_size, f_size, bias)
        self.layer3 = SCConv_Layer(f_size, f_size, f_size, f_size, bias)

        self.combined1 = nn.Linear(3 * f_size, output_size, bias)
        self.combined2 = nn.Linear(3 * f_size, output_size, bias)
        self.combined3 = nn.Linear(3 * f_size, output_size, bias)

        self.output = nn.Linear(output_size * 3, output_size, bias)

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, L1, L2 = simplicialComplex.unpack_laplacians()
        batch = simplicialComplex.unpack_batch()
        B2D3, D2B1TD1inv, D1invB1, B2TD2inv = simplicialComplex.unpack_others()

        X0_0, X1_0, X2_0 = self.layer1(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
        X0_1, X1_1, X2_1 = self.layer2(X0_0, X1_0, X2_0, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
        X0_2, X1_2, X2_2 = self.layer3(X0_1, X1_1, X2_1, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)

        X0 = self.combined1(torch.cat([X0_0, X0_1, X0_2], dim=1))
        X1 = self.combined2(torch.cat([X1_0, X1_1, X1_2], dim=1))
        X2 = self.combined3(torch.cat([X2_0, X2_1, X2_2], dim=1))

        X0 = global_mean_pool(X0, batch[0])
        X1 = global_mean_pool(X1, batch[1])
        X2 = global_mean_pool(X2, batch[2])

        out = torch.cat([X0, X1, X2], dim=1)

        return F.softmax(self.output(out), dim=1)


class PRELU(nn.PReLU):

    def forward(self, input):
        return F.prelu(input, self.weight)


class PlanetoidSCConv(nn.Module):

    def __init__(self, num_node_feats, output_size, bias=True):
        super().__init__()
        f_size = output_size
        self.layer1 = SCConv_Layer(num_node_feats, num_node_feats, num_node_feats, f_size, bias, PRELU())

        self.tri_layer = nn.Linear(output_size, output_size)

    def forward(self, simplicialComplex, B1, B2):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, L1, L2 = simplicialComplex.unpack_laplacians()
        B2D3, D2B1TD1inv, D1invB1, B2TD2inv = simplicialComplex.unpack_others()

        X0[X0 != 0] = 1

        X1_in, X1_out = X0[X1[:, 0]], X0[X1[:, 1]]
        X1 = torch.logical_and(X1_in, X1_out).float()

        X2_i, X2_j, X2_k = X0[X2[:, 0]], X0[X2[:, 1]], X0[X2[:, 2]]
        X2 = torch.logical_and(X2_i, torch.logical_and(X2_j, X2_k)).float()

        X0, X1, X2 = self.layer1(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
        X0 = (X0 + torch.sparse.mm(B1, X1) + torch.sparse.mm(B1, self.tri_layer(torch.sparse.mm(B2, X2)))) / 3
        return X0


class FlowSCConv(nn.Module):

    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=False, f=F.relu):
        super().__init__()
        f_size = 32

        self.layer1 = SCConv_Layer(num_node_feats, num_edge_feats, num_triangle_feats, f_size, bias=bias, f=f)
        self.layer2 = SCConv_Layer(f_size, f_size, f_size, f_size, bias=bias, f=f)
        self.layer3 = SCConv_Layer(f_size, f_size, f_size, f_size, bias=bias, f=f)
        self.layer4 = SCConv_Layer(f_size, f_size, f_size, f_size, bias=bias, f=f)
        self.mlp1 = nn.Linear(f_size, f_size)
        self.mlp2 = nn.Linear(f_size, output_size)

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, L1, L2 = simplicialComplex.unpack_laplacians()
        batch = simplicialComplex.unpack_batch()
        B2D3, D2B1TD1inv, D1invB1, B2TD2inv = simplicialComplex.unpack_others()

        X0, X1, X2 = self.layer1(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
        X0, X1, X2 = self.layer2(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
        X0, X1, X2 = self.layer3(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
        _, X1, _ = self.layer4(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)

        X1 = global_mean_pool(X1.abs(), batch[1])
        X1 = F.relu(self.mlp1(X1))

        return torch.softmax(self.mlp2(X1), dim=1)


class TestSCConv(nn.Module):

    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=False, f=F.relu):
        super().__init__()
        self.layer1 = SCConv_Layer(num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=bias, f=f)

    def forward(self, simplicialComplex):
        X0, X1, X2 = simplicialComplex.unpack_features()
        L0, L1, L2 = simplicialComplex.unpack_laplacians()
        B2D3, D2B1TD1inv, D1invB1, B2TD2inv = simplicialComplex.unpack_others()

        X0, X1, X2 = self.layer1(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)

        return X0, X1, X2