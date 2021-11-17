import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from models.ProcessorTemplate import GNNTemplate, GNNBatcher, GNNPreprocessor
from models.nn_utils import normalise, collated_data_to_batch,\
    convert_indices_and_values_to_sparse, torch_sparse_to_scipy_sparse
from scipy import sparse
import numpy as np

class SNN_Bunch(GNNTemplate):
  
    def __init__(self, f1_size, f2_size, f3_size, output_size):
        super().__init__(output_size)

    def forward(self, features_dct):
      return


    def _normalised_scData_to_Lapacian(self, scData):

        def to_dense(matrix):
            indices = matrix[0:2]
            values = matrix[2:3].squeeze()
            return torch.sparse_coo_tensor(indices, values).to_dense()

        B1, B2 = to_dense(scData.b1), to_dense(scData.b2)
        X0, X1, X2 = scData.X0, scData.X1, scData.X2

        L0 = B1 @ B1.T
        B1_sum = torch.sum(torch.abs(B1), 1)
        d0 = torch.diag(B1_sum)
        B1_sum_inv = torch.nan_to_num(1. / B1_sum, nan=0., posinf=0., neginf=0.)
        d0_inv = torch.diag(B1_sum_inv)
        L0 = d0_inv @ L0
        L0_factor = -1 * torch.diag(1 / (B1_sum_inv + 1))
        L0bias = torch.eye(n=d0.shape[0])
        L0 = L0_factor @ L0 + L0bias

        D1_inv = torch.diag(0.5 * B1_sum_inv)
        D2diag = torch.sum(torch.abs(B2), 1)
        D2diag = torch.maximum(D2diag, torch.tensor([1 for _ in range(D2diag.shape[0])]))
        D2 = torch.diag(D2diag)
        D2_inv = torch.diag(1 / D2diag)
        D3 = (1 / 3.) * torch.eye(n=B2.shape[1])

        # might need to change this
        A_1u = D2 - B2 @ D3 @ B2.T
        A_1d = D2_inv - B1.T @ D1_inv @ B1
        A_1u_norm = (A_1u + torch.eye(n = A_1u.shape[0])) @ (torch.diag(1/(D2diag + 1)))
        A_1d_norm = (D2 + torch.eye(n=D2.shape[0])) @ (A_1d + torch.eye(n=A_1d.shape[0]))
        L1 = A_1u_norm + A_1d_norm
        
        B2_sum = torch.sum(torch.abs(B2), 1)
        B2_sum_inv = 1 / (B2_sum + 1)
        D5inv = torch.diag(B2_sum_inv)

        A_2d = torch.eye(n=B2.shape[1]) + B2.T @ D5inv @ B2
        A_2d_norm = (2 * torch.eye(n=B2.shape[1])) @ (A_2d + torch.eye(n=A_2d.shape[0]))
        L2 = A_2d_norm  # normalized adjacency

        B2D3 = B2 @ D3
        D2B1TD1inv = (1 / np.sqrt(2.)) * D2 @ B1.T @ D1_inv
        D1invB1 = (1 / np.sqrt(2.)) * D1_inv @ B1
        B2TD2inv = B2.T @ D5inv

        L0, L1, L2 = L0.to_sparse(), L1.to_sparse(), L2.to_sparse()
        B2D3, D2B1TD1inv, D1invB1, B2TD2inv = B2D3.to_sparse(), D2B1TD1inv.to_sparse(),\
                                              D1invB1.to_sparse(), B2TD2inv.to_sparse()


    def batch(self, scDataList):
        features_dct, label = collated_data_to_batch(scDataList, self._normalised_scData_to_Lapacian)
        return features_dct, label

    def clean_feature_dct(self, feature_dct):
        return convert_indices_and_values_to_sparse(feature_dct)