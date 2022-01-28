import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from models.nn_utils import chebyshev, unpack_feature_dct_to_L_X_B


class SCNLayer(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias = True, k = 1):
        super().__init__()
        self.k = k
        self.conv = nn.Linear(k * feature_size, output_size, bias = enable_bias)

    def forward(self, L, x):
        X = chebyshev(L, x, self.k)
        return self.conv(X)


class SuperpixelEbli(nn.Module):
    # This model is based on model described by Stefanie Ebli et al. in Simplicial Neural Networks
    # Github here https://github.com/stefaniaebli/simplicial_neural_networks?utm_source=catalyzex.com
    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats, output_size, bias = True):
        super().__init__()

        conv_size = 32

        # Degree 0 convolutions.
        self.C0_1 = SCNLayer(num_node_feats, conv_size, enable_bias = bias)
        self.C0_2 = SCNLayer(conv_size, conv_size, enable_bias = bias)
        self.C0_3 = SCNLayer(conv_size, conv_size, enable_bias = bias)

        # Degree 1 convolutions.
        self.C1_1 = SCNLayer(num_edge_feats, conv_size, enable_bias = bias)
        self.C1_2 = SCNLayer(conv_size, conv_size, enable_bias = bias)
        self.C1_3 = SCNLayer(conv_size, conv_size, enable_bias = bias)

        # Degree 2 convolutions.
        self.C2_1 = SCNLayer(num_triangle_feats, conv_size, enable_bias = bias)
        self.C2_2 = SCNLayer(conv_size, conv_size, enable_bias = bias)
        self.C2_3 = SCNLayer(conv_size, conv_size, enable_bias = bias)

        self.layer0 = nn.Linear(3 * conv_size, output_size)
        self.layer1 = nn.Linear(3 * conv_size, output_size)
        self.layer2 = nn.Linear(3 * conv_size, output_size)

        self.combined_layer = nn.Linear(output_size * 3, output_size)


    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        out0_1 = self.C0_1(L[0], X[0])
        out0_2 = self.C0_2(L[0], nn.LeakyReLU()(out0_1))
        out0_3 = self.C0_3(L[0], nn.LeakyReLU()(out0_2))
        out0 = self.layer0(torch.cat([out0_1, out0_2, out0_3], dim = 1))
        out0 = global_mean_pool(out0, batch[0])

        out1_1 = self.C1_1(L[1], X[1])
        out1_2 = self.C1_2(L[1], nn.LeakyReLU()(out1_1))
        out1_3 = self.C1_3(L[1], nn.LeakyReLU()(out1_2))
        out1 = self.layer1(torch.cat([out1_1, out1_2, out1_3], dim=1))
        out1 = global_mean_pool(out1, batch[1])

        out2_1 = self.C2_1(L[2], X[2])
        out2_2 = self.C2_2(L[2], nn.LeakyReLU()(out2_1))
        out2_3 = self.C2_3(L[2], nn.LeakyReLU()(out2_2))
        out2 = self.layer2(torch.cat([out2_1, out2_2, out2_3], dim=1))
        out2 = global_mean_pool(out2, batch[2])

        # return F.softmax((out0 + out1 + out2)/3)
        out = torch.cat([out0, out1, out2], dim = 1)
        return F.softmax(self.combined_layer(out), dim = 1)
