import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_scatter import scatter_max


class SCN(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias = True):
        super().__init__()
        self.conv = nn.Linear(feature_size, output_size)
        # if enable_bias:
        #     self.bias = nn.parameter.Parameter(torch.zeros((1, output_size, 1)))
        # else:
        #     self.bias = 0.0

    def forward(self, L, x):
        x = torch.sparse.mm(L, x)
        return self.conv(x)


class SNN(nn.Module):
    def __init__(self, f1_size, f2_size, f3_size, output_size):
        super().__init__()

        # Degree 0 convolutions.
        self.C0_1 = SCN(f1_size, 32)
        self.C0_2 = SCN(32, 32)
        self.C0_3 = SCN(32, output_size)

        # Degree 1 convolutions.
        self.C1_1 = SCN(f2_size, 32)
        self.C1_2 = SCN(32, 32)
        self.C1_3 = SCN(32, output_size)

        # Degree 2 convolutions.
        self.C2_1 = SCN(f3_size, 32)
        self.C2_2 = SCN(32, 32)
        self.C2_3 = SCN(32, output_size)

        self.output_size = output_size


    def forward(self, X0, X1, X2, L0, L1, L2, batch0, batch1, batch2):

        out0_1 = self.C0_1(L0, nn.LeakyReLU()(X0))
        out0_2 = self.C0_2(L0, nn.LeakyReLU()(out0_1))
        out0_3 = self.C0_3(L0, out0_2)

        # out0 = global_max_pool(out0_3, batch0)

        out1_1 = self.C1_1(L1, nn.LeakyReLU()(X1))
        out1_2 = self.C1_2(L1, nn.LeakyReLU()(out1_1))
        out1_3 = self.C1_3(L1, out1_2)

        # out1 = global_max_pool(out1_3, batch1)

        out2_1 = self.C2_1(L2, nn.LeakyReLU()(X2))
        out2_2 = self.C2_2(L2, nn.LeakyReLU()(out2_1))
        out2_3 = self.C2_3(L2, out2_2)

        # out2 = global_max_pool(out2_3, batch2)

        out = torch.cat([out0_3, out1_3, out2_3], dim=0)
        out = torch.sum(out, 0)
        return F.softmax(out, dim = 0).view(1, self.output_size)
