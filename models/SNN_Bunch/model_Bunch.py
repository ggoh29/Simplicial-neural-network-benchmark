import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from models.nn_utils import unpack_feature_dct_to_L_X_B, preprocess_features

class SNN_Bunch_Layer(nn.Module):
  # This model is based on model described by Eric Bunch et al. in Simplicial 2-Complex Convolutional Neural Networks
  # Github here https://github.com/AmFamMLTeam/simplicial-2-complex-cnns
  def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True, f = F.relu):

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

    X0 = self.w(n2n)
    # X0 = (1 / 2.) * self.w(n2n + e2n)
    X1 = (1 / 3.) * self.w(e2e + n2e + t2e)
    X2 = (1 / 2.) * self.w(t2t + e2t)

    return X0, X1, X2


class SuperpixelBunch(nn.Module):

  def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True):

    super().__init__()
    f_size = 20
    self.layer1 = SNN_Bunch_Layer(num_node_feats, num_edge_feats, num_triangle_feats, f_size, bias)
    self.layer2 = SNN_Bunch_Layer(f_size, f_size, f_size, f_size, bias)
    self.layer3 = SNN_Bunch_Layer(f_size, f_size, f_size, f_size, bias)

    self.combined1 = nn.Linear(3 * f_size, output_size)
    self.combined2 = nn.Linear(3 * f_size, output_size)
    self.combined3 = nn.Linear(3 * f_size, output_size)

    self.output = nn.Linear(output_size * 3, output_size, bias)


  def forward(self, feature_dct):
    L, X, batch = unpack_feature_dct_to_L_X_B(feature_dct)
    B2D3, D2B1TD1inv, D1invB1, B2TD2inv = feature_dct['others']

    X0, X1, X2 = X
    L0, L1, L2 = L

    X0_0, X1_0, X2_0 = self.layer1(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
    X0_1, X1_1, X2_1 = self.layer2(X0_0, X1_0, X2_0, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
    X0_2, X1_2, X2_2 = self.layer3(X0_1, X1_1, X2_1, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)

    X0 = self.combined1(torch.cat([X0_0, X0_1, X0_2], dim = 1))
    X1 = self.combined1(torch.cat([X1_0, X1_1, X1_2], dim=1))
    X2 = self.combined1(torch.cat([X2_0, X2_1, X2_2], dim=1))

    X0 = global_mean_pool(X0, batch[0])
    X1 = global_mean_pool(X1, batch[1])
    X2 = global_mean_pool(X2, batch[2])

    out = torch.cat([X0, X1, X2], dim=1)

    return F.softmax(self.output(out), dim=1)

class PRELU(nn.PReLU):

  def forward(self, input):
    return F.prelu(input, self.weight)


class SNN_b(nn.Module):
  # This model is based on model described by Eric Bunch et al. in Simplicial 2-Complex Convolutional Neural Networks
  # Github here https://github.com/AmFamMLTeam/simplicial-2-complex-cnns
  def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias=True, f = F.relu):

    super().__init__()
    self.n2n_weights = nn.Linear(num_node_feats, output_size, bias=bias)
    self.n2e_weights = nn.Linear(num_node_feats, output_size, bias=bias)
    self.e2e_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
    self.e2n_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
    self.e2t_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
    self.w = f


  def forward(self, X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv):

    n2n = self.n2n_weights(X0)  # Y00 = X0*W00
    n2n = torch.sparse.mm(L0, n2n)  # L0*Y00

    n2e = self.n2e_weights(X0)  # Y01 = X0*W01
    n2e = torch.sparse.mm(D2B1TD1inv, n2e)  # D2*B1.T*D1^-1*Y01

    e2n = self.e2n_weights(X1)  # Y10 = X1*W10
    e2n = torch.sparse.mm(D1invB1, e2n)  # D1invB1*Y10

    e2e = self.e2e_weights(X1)  # Y11 = X1*W11
    e2e = torch.sparse.mm(L1, e2e)  # L1*Y11
    #
    # e2t = self.e2t_weights(X1)  # Y21 = X1*W21
    # e2t = torch.sparse.mm(B2TD2inv, e2t)  # B2TD2inv*Y21

    X0 = (1 / 2.) * self.w(n2n + e2n)
    X1 = (1 / 2.) *  self.w(e2e + n2e)
    # X2 = self.w(e2t)

    return X0, X1, X2


class PlanetoidBunch(nn.Module):

  def __init__(self, num_node_feats, output_size, bias=True):

    super().__init__()
    f_size = output_size
    self.layer1 = SNN_Bunch_Layer(num_node_feats, num_node_feats, num_node_feats, f_size, bias, PRELU())

    self.tri_layer = nn.Linear(output_size, output_size)
    self.final_layer = nn.Linear(3 * output_size, output_size)
    # self.w = PRELU()


  def forward(self, feature_dct):
    L, X, batch = unpack_feature_dct_to_L_X_B(feature_dct)
    B2D3, D2B1TD1inv, D1invB1, B2TD2inv = feature_dct['others']

    X0, X1, X2 = X
    L0, L1, L2 = L

    X1_in, X1_out = X0[X1[:, 0]], X0[X1[:, 1]]
    X1 = torch.logical_and(X1_in, X1_out).float()
    #
    X2_i, X2_j, X2_k = X0[X2[:, 0]], X0[X2[:, 1]], X0[X2[:, 2]]
    X2 = torch.logical_and(X2_i, torch.logical_and(X2_j, X2_k)).float()

    # X1 = torch.ones((X1.shape[0], X0.shape[1])).cuda()
    # X2 = torch.ones((X2.shape[0], X0.shape[1])).cuda()

    X0, X1, X2 = self.layer1(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv)
    # Y0_cat = torch.cat([X0, torch.sparse.mm(D1invB1, X1)], dim = 1)
    # Y0_cat = torch.cat([X0, torch.sparse.mm(D1invB1, X1), torch.sparse.mm(D1invB1, self.tri_layer(torch.sparse.mm(B2D3, X2)))], dim = 1)
    # X0 = self.final_layer(Y0_cat)
    # X0 = (X0 + torch.sparse.mm(D1invB1, X1)) / 2
    # X0 = (X0 + torch.sparse.mm(D1invB1, X1) + torch.sparse.mm(D1invB1, torch.sparse.mm(B2D3, X2))) / 3
    X0 = X0 + torch.sparse.mm(D1invB1, X1) + torch.sparse.mm(D1invB1, self.tri_layer(torch.sparse.mm(B2D3, X2)))/3
    return X0