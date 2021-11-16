import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from models.GNN_template import GCNTemplate
from models.nn_utils import normalise, collated_data_to_batch,\
    convert_indices_and_values_to_sparse, torch_sparse_to_scipy_sparse
from scipy import sparse
import numpy as np

class SNN_Bunch(GCNTemplate):
  
    def __init__(self, f1_size, f2_size, f3_size, output_size):
        super().__init__(output_size)

    def forward(self, features_dct):
      return


    def _normalised_scData_to_Lapacian(self, scData):

        def to_dense(matrix):
            indices = matrix[0:2]
            values = matrix[2:3].squeeze()
            return torch.sparse_coo_tensor(indices, values)

        b1, b2 = to_dense(scData.b1), to_dense(scData.b2)

        B1, B2 = torch_sparse_to_scipy_sparse(b1), torch_sparse_to_scipy_sparse(b2)

        X0, X1, X2 = scData.X0, scData.X1, scData.X2

        b1_d, b2_d = b1.to_dense(), b2.to_dense()
        L0 = torch.matmul(b1_d, b1_d.T)
        b1_sum = torch.sum(torch.abs(b1_d), 1)
        d0 = torch.diag(b1_sum)
        b1_sum_inv = torch.nan_to_num(1. / b1_sum, nan=0., posinf=0., neginf=0.)
        d0_inv = torch.diag(b1_sum_inv)
        L0 = torch.matmul(d0_inv, L0)
        L0_factor = -1 * torch.diag(1 / (b1_sum_inv + 1))
        L0bias = torch.eye(n=d0.shape[0])
        L0 = torch.matmul(L0_factor, L0) + L0bias


        L0 = B1 @ B1.T
        B1_sum = np.abs(B1).sum(axis=1)
        D0 = sparse.diags(B1_sum.A.reshape(-1), 0)
        B1_sum_inv = 1. / B1_sum
        B1_sum_inv[np.isinf(B1_sum_inv) | np.isneginf(B1_sum_inv)] = 0
        D0_inv = sparse.diags(B1_sum_inv.A.reshape(-1), 0)
        L0 = D0_inv @ L0
        L0factor = (-1) * sparse.diags((1 / (B1_sum_inv + 1)).A.reshape(-1), 0)
        L0bias = sparse.identity(n=D0.shape[0])
        L0 = L0factor @ L0 + L0bias

        D1_inv = sparse.diags((B1_sum_inv * 0.5).A.reshape(-1), 0)
        D2diag = abs(B2).sum(axis=1).A.reshape(-1)
        D2diag = np.max(
            (
                D2diag,
                np.ones(shape=D2diag.shape[0])
            ),
            axis=0
        )
        D2 = sparse.diags(D2diag, 0)
        D2inv = sparse.diags(1 / D2diag, 0)
        D3 = (1 / 3.) * sparse.identity(n=B2.shape[1])

        A_1u = D2 - B2 @ D3 @ B2.T
        A_1d = D2inv - B1.T @ D1_inv @ B1
        A_1u_norm = (
                        A_1u
                        +
                        sparse.identity(n=A_1u.shape[0])
                    ) @ sparse.diags(1 / (D2.diagonal() + 1), 0)
        A_1d_norm = (
                        D2
                        +
                        sparse.identity(n=D2.shape[0])
                    ) @ (A_1d + sparse.identity(n=A_1d.shape[0]))
        # not really L1, but easy to drop in; normalized adjacency
        L1 = A_1u_norm + A_1d_norm

        B2_sum = abs(B2).sum(axis=1)
        B2_sum_inv = 1 / (B2_sum + 1)
        D5inv = sparse.diags(B2_sum_inv, 0)

        A_2d = sparse.identity(n=B2.shape[1]) + B2.T @ D5inv @ B2
        A_2d_norm = (2 * sparse.identity(n=B2.shape[1])) @ (
            A_2d
            +
            sparse.identity(n=A_2d.shape[0])
        )
        L2 = A_2d_norm  # normalized adjacency

        B2D3 = B2 @ D3
        D2B1TD1inv = (1 / np.sqrt(2.)) * D2 @ B1.T @ D1_inv
        D1invB1 = (1 / np.sqrt(2.)) * D1_inv @ B1
        B2TD2inv = B2.T @ D5inv

        print(L0, L1, L2)


    def batch(self, scDataList):
        features_dct, label = collated_data_to_batch(scDataList, self._normalised_scData_to_Lapacian)
        return features_dct, label

    def clean_feature_dct(self, feature_dct):
        return convert_indices_and_values_to_sparse(feature_dct)