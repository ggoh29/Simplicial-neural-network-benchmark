import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F


class SCN1(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias = True):
        super().__init__()
        self.conv = nn.Linear(feature_size, output_size, bias = enable_bias)

    def forward(self, L, x):
        x = torch.sparse.mm(L, x)
        return self.conv(x)


class SCN(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias = True):
        super().__init__()
        self.theta = nn.parameter.Parameter(0.01 * torch.randn((feature_size, output_size)))

    def forward(self, L, x):
        X = torch.sparse.mm(L, x)
        return torch.mm(X, self.theta)


class SNN(nn.Module):
    def __init__(self, f1_size, f2_size, f3_size, output_size, bias = True):
        super().__init__()

        conv_size = 15

        # Degree 0 convolutions.
        self.C0_1 = SCN(f1_size, conv_size, enable_bias = bias)
        self.C0_2 = SCN(conv_size, conv_size, enable_bias = bias)
        self.C0_3 = SCN(conv_size, output_size, enable_bias = bias)

        # Degree 1 convolutions.
        self.C1_1 = SCN(f2_size, conv_size, enable_bias = bias)
        self.C1_2 = SCN(conv_size, conv_size, enable_bias = bias)
        self.C1_3 = SCN(conv_size, output_size, enable_bias = bias)

        # Degree 2 convolutions.
        self.C2_1 = SCN(f3_size, conv_size, enable_bias = bias)
        self.C2_2 = SCN(conv_size, conv_size, enable_bias = bias)
        self.C2_3 = SCN(conv_size, output_size, enable_bias = bias)

        self.layer = nn.Linear(output_size, output_size)
        self.output_size = output_size


    def forward(self, X, L, batch):

        out0_1 = self.C0_1(L[0], nn.LeakyReLU()(X[0]))
        out0_2 = self.C0_2(L[0], nn.LeakyReLU()(out0_1))
        out0_3 = self.C0_3(L[0], out0_2)

        out0 = global_mean_pool(out0_3, batch[0])

        out1_1 = self.C1_1(L[1], nn.LeakyReLU()(X[1]))
        out1_2 = self.C1_2(L[1], nn.LeakyReLU()(out1_1))
        out1_3 = self.C1_3(L[1], out1_2)

        out1 = global_mean_pool(out1_3, batch[1])

        out2_1 = self.C2_1(L[2], nn.LeakyReLU()(X[2]))
        out2_2 = self.C2_2(L[2], nn.LeakyReLU()(out2_1))
        out2_3 = self.C2_3(L[2], out2_2)

        out2 = global_mean_pool(out2_3, batch[2])

        return F.softmax(self.layer(out0 + out1 + out2), dim = 1)