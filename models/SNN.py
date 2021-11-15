import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from models.GNN_template import GCNTemplate
from models.nn_utils import chebyshev, normalise, collated_data_to_batch,\
    unpack_feature_dct_to_L_X_B, convert_indices_and_values_to_sparse


class SCN(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias = True, k = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Linear(k * feature_size, output_size, bias = enable_bias)

    def forward(self, L, x):
        X = chebyshev(L, x, self.k)
        return self.conv(X)


class SCN1(nn.Module):
    def __init__(self, feature_size, output_size, enable_bias = True, k = 3):
        super().__init__()
        self.k = k
        self.theta = nn.parameter.Parameter(0.01 * torch.randn((k * feature_size, output_size)))

    def forward(self, L, x):
        # X = chebyshev(L, x, self.k)
        X = torch.sparse.mm(L, x)
        return torch.mm(X, self.theta)


class SNN(GCNTemplate):
    def __init__(self, f1_size, f2_size, f3_size, output_size, bias = True):
        super().__init__(output_size)

        conv_size = 32

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

        self.layer = nn.Linear(output_size * 3, output_size)


    def forward(self, features_dct):
        L, X, batch = unpack_feature_dct_to_L_X_B(features_dct)

        out0_1 = self.C0_1(L[0], X[0])
        out0_2 = self.C0_2(L[0], nn.LeakyReLU()(out0_1))
        out0_3 = self.C0_3(L[0], nn.LeakyReLU()(out0_2))

        out0 = global_mean_pool(out0_3, batch[0])

        out1_1 = self.C1_1(L[1], X[1])
        out1_2 = self.C1_2(L[1], nn.LeakyReLU()(out1_1))
        out1_3 = self.C1_3(L[1], nn.LeakyReLU()(out1_2))

        out1 = global_mean_pool(out1_3, batch[1])

        out2_1 = self.C2_1(L[2], X[2])
        out2_2 = self.C2_2(L[2], nn.LeakyReLU()(out2_1))
        out2_3 = self.C2_3(L[2], nn.LeakyReLU()(out2_2))

        out2 = global_mean_pool(out2_3, batch[2])

        out = torch.cat([out0, out1, out2], dim = 1)

        return F.softmax(self.layer(out), dim = 1)


    def _normalised_scData_to_Lapacian(self, scData):

        def to_sparse_coo(matrix):
            indices = matrix[0:2]
            values = matrix[2:3].squeeze()
            return torch.sparse_coo_tensor(indices, values)

        sigma1, sigma2 = to_sparse_coo(scData.sigma1), to_sparse_coo(scData.sigma2)

        X0, X1, X2 = scData.X0, scData.X1, scData.X2
        L0 = normalise(torch.sparse.mm(sigma1, sigma1.t()))
        L1 = normalise(torch.sparse.FloatTensor.add(torch.sparse.mm(sigma1.t(), sigma1), torch.sparse.mm(sigma2, sigma2.t())))
        L2 = normalise(torch.sparse.mm(sigma2.t(), sigma2))

        # splitting the sparse tensor as pooling cannot return sparse and to make preparation for minibatching easier
        assert (X0.size()[0] == L0.size()[0])
        assert (X1.size()[0] == L1.size()[0])
        assert (X2.size()[0] == L2.size()[0])

        return [[X0, X1, X2],
                [L0.coalesce().indices(), L1.coalesce().indices(), L2.coalesce().indices()],
                [L0.coalesce().values(), L1.coalesce().values(), L2.coalesce().values()]], scData.label


    def batch(self, scDataList):
        features_dct, label = collated_data_to_batch(scDataList, self._normalised_scData_to_Lapacian)
        return features_dct, label

    def clean_feature_dct(self, feature_dct):
        return convert_indices_and_values_to_sparse(feature_dct)