from utils import ensure_input_is_tensor
import torch
import numpy as np
from models.ProcessorTemplate import NNProcessor
from models.nn_utils import convert_indices_and_values_to_sparse, \
    batch_sparse_matrix, batch_all_feature_and_lapacian_pair
from models.nn_utils import  unpack_feature_dct_to_L_X_B, repair_sparse, sparse_diag_identity, sparse_diag


def compute_d2(B):
    D2diag = torch.maximum(torch.diag(B), torch.ones(B.shape[0]))
    return torch.diag(D2diag)


def compute_d1(B, D):
    B_v_abs, B_i = torch.abs(B.coalesce().values()), B.coalesce().indices()
    B_abs = torch.sparse_coo_tensor(B_i, B_v_abs)
    diag = torch.sparse.sum(torch.sparse.mm(B_abs, D), dim=1).to_dense()
    return 2 * torch.diag(diag)


class SimplicialObject:

    def __init__(self, X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv, label):
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2

        self.L0 = ensure_input_is_tensor(L0)
        self.L1 = ensure_input_is_tensor(L1)
        self.L2 = ensure_input_is_tensor(L2)

        self.B2D3 = ensure_input_is_tensor(B2D3)
        self.D2B1TD1inv = ensure_input_is_tensor(D2B1TD1inv)
        self.D1invB1 = ensure_input_is_tensor(D1invB1)
        self.B2TD2inv = ensure_input_is_tensor(B2TD2inv)

        self.label = label

    def __eq__(self, other):
        x0 = torch.allclose(self.X0, other.X0, atol=1e-5)
        x1 = torch.allclose(self.X1, other.X1, atol=1e-5)
        x2 = torch.allclose(self.X2, other.X2, atol=1e-5)
        l0 = torch.allclose(self.L0, other.L0, atol=1e-5)
        l1 = torch.allclose(self.L1, other.L1, atol=1e-5)
        l2 = torch.allclose(self.L2, other.L2, atol=1e-5)
        d1 = torch.allclose(self.B2D3, other.B2D3, atol=1e-5)
        d2 = torch.allclose(self.D2B1TD1inv, other.D2B1TD1inv, atol=1e-5)
        d3 = torch.allclose(self.D1invB1, other.D1invB1, atol=1e-5)
        d4 = torch.allclose(self.B2TD2inv, other.B2TD2inv, atol=1e-5)
        label = torch.allclose(self.label, other.label, atol=1e-5)
        return all([x0, x1, x2, l0, l1, l2, d1, d2, d3, d4, label])


class SNNBunchProcessor(NNProcessor):

    # This model is based on model described by Eric Bunch et al. in Simplicial 2-Complex Convolutional Neural Networks
    # Github here https://github.com/AmFamMLTeam/simplicial-2-complex-cnns

    def _process(self, scData):
        def to_sparse(matrix, size):
            indices = matrix[0:2]
            values = matrix[2:3].squeeze()
            return torch.sparse_coo_tensor(indices, values, size)

        X0, X1, X2 = scData.X0.cpu(), scData.X1.cpu(), scData.X2.cpu()
        label = scData.label

        x0, x1, x2 = X0.shape[0], X1.shape[0], X2.shape[0]
        B1, B2 = to_sparse(scData.b1, (x0, x1)).cpu(), to_sparse(scData.b2, (x1, x2)).cpu()

        L0 = torch.sparse.mm(B1, B1.t())
        B1_v_abs, B1_i = torch.abs(B1.coalesce().values()), B1.coalesce().indices()
        B1_sum = torch.sparse.sum(torch.sparse_coo_tensor(B1_i, B1_v_abs, (x0, x1)), dim=1)
        B1_sum_values = B1_sum.to_dense()
        B1_sum_values = torch.maximum(B1_sum_values, torch.ones(B1_sum_values.shape[0]))
        B1_sum_indices = torch.tensor([i for i in range(x0)])
        d0_diag_indices = torch.stack([B1_sum_indices, B1_sum_indices], dim=0)
        d0 = torch.sparse_coo_tensor(d0_diag_indices, B1_sum_values, (x0, x0))
        B1_sum_inv_values = torch.nan_to_num(1. / B1_sum_values, nan=0., posinf=0., neginf=0.)
        L0 = torch.sparse.mm(L0, d0)

        D1_inv = torch.sparse_coo_tensor(d0_diag_indices, 0.5 * B1_sum_inv_values, (x0, x0))
        B2_v_abs, B2_i = torch.abs(B2.coalesce().values()), B2.coalesce().indices()
        D2diag_1 = torch.sparse.sum(torch.sparse_coo_tensor(B2_i, B2_v_abs, (x1, x1)), dim=1).to_dense()
        D2diag = torch.maximum(D2diag_1, torch.ones(D2diag_1.shape[0]))
        D2_indices = [i for i in range(D2diag.shape[0])]
        D2_indices = torch.tensor([D2_indices, D2_indices])
        D2 = torch.sparse_coo_tensor(D2_indices, D2diag, (x1, x1))
        D2_inv = torch.sparse_coo_tensor(D2_indices, 1 / D2diag, (x1, x1))
        D3_values = (1 / 3.) * torch.ones(B2.shape[1])
        D3_indices = [i for i in range(B2.shape[1])]
        D3_indices = torch.tensor([D3_indices, D3_indices])
        D3 = torch.sparse_coo_tensor(D3_indices, D3_values, (x2, x2))

        L_1u = torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(B2, D3), B2.t()), D2_inv)
        L_1d = torch.sparse.mm(D2, torch.sparse.mm(torch.sparse.mm(B1.t(), D1_inv), B1))

        L1 = L_1u + L_1d

        B2_sum = D2diag_1
        B2_sum_inv = 1 / (B2_sum + 1)
        D5inv = sparse_diag(B2_sum_inv)
        L2 = torch.sparse.mm(torch.sparse.mm(B2.t(), D5inv), B2)

        B2D3 = torch.sparse.mm(B2, D3)
        D2B1TD1inv = (1 / np.sqrt(2.)) * torch.sparse.mm(torch.sparse.mm(D2, B1.t()), D1_inv)
        D1invB1 = (1 / np.sqrt(2.)) * torch.sparse.mm(D1_inv, B1)
        B2TD2inv = torch.sparse.mm(B2.t(), D5inv)

        # L0 = (torch.eye(x0) - L0).to_sparse()
        # L1 = (torch.eye(x1) - L1).to_sparse()
        # L2 = (torch.eye(x2) - L2).to_sparse()

        return SimplicialObject(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv, label)

    def process(self, scData):
        # This does processing in a sparse format
        def to_sparse(matrix, size):
            indices = matrix[0:2]
            values = matrix[2:3].squeeze()
            return torch.sparse_coo_tensor(indices, values, size)

        X0, X1, X2 = scData.X0.cpu(), scData.X1.cpu(), scData.X2.cpu()
        label = scData.label

        x0, x1, x2 = X0.shape[0], X1.shape[0], X2.shape[0]
        B1, B2 = to_sparse(scData.b1, (x0, x1)).cpu(), to_sparse(scData.b2, (x1, x2)).cpu()

        L0 = torch.sparse.mm(B1, B1.t())
        B1_v_abs, B1_i = torch.abs(B1.coalesce().values()), B1.coalesce().indices()
        B1_sum = torch.sparse.sum(torch.sparse_coo_tensor(B1_i, B1_v_abs, (x0, x1)), dim=1)
        B1_sum_values = B1_sum.to_dense()
        B1_sum_indices = torch.tensor([i for i in range(x0)])
        d0_diag_indices = torch.stack([B1_sum_indices, B1_sum_indices], dim=0)
        d0 = torch.sparse_coo_tensor(d0_diag_indices, B1_sum_values, (x0, x0))
        B1_sum_inv_values = torch.nan_to_num(1. / B1_sum_values, nan=0., posinf=0., neginf=0.)
        d0_inv = torch.sparse_coo_tensor(d0_diag_indices, B1_sum_inv_values, (x0, x0))
        L0 = torch.sparse.mm(d0_inv, L0)
        L0_factor_values = -1 / (B1_sum_inv_values + 1)
        L0_factor = torch.sparse_coo_tensor(d0_diag_indices, L0_factor_values, (x0, x0))
        L0_bias_values = torch.ones(d0.shape[0])
        L0bias = torch.sparse_coo_tensor(d0_diag_indices, L0_bias_values, (x0, x0))
        L0 = L0bias + torch.sparse.mm(L0_factor, L0)

        D1_inv = torch.sparse_coo_tensor(d0_diag_indices, 0.5 * B1_sum_inv_values, (x0, x0))
        B2_v_abs, B2_i = torch.abs(B2.coalesce().values()), B2.coalesce().indices()
        D2diag_1 = torch.sparse.sum(torch.sparse_coo_tensor(B2_i, B2_v_abs, (x1, x1)), dim=1).to_dense()
        D2diag = torch.maximum(D2diag_1, torch.ones(D2diag_1.shape[0]))
        D2_indices = [i for i in range(D2diag.shape[0])]
        D2_indices = torch.tensor([D2_indices, D2_indices])
        D2 = torch.sparse_coo_tensor(D2_indices, D2diag, (x1, x1))
        D2_inv = torch.sparse_coo_tensor(D2_indices, 1 / D2diag, (x1, x1))
        D3_values = (1 / 3.) * torch.ones(B2.shape[1])
        D3_indices = [i for i in range(B2.shape[1])]
        D3_indices = torch.tensor([D3_indices, D3_indices])
        D3 = torch.sparse_coo_tensor(D3_indices, D3_values, (x2, x2))
        A_1u = D2 - torch.sparse.mm(torch.sparse.mm(B2, D3), B2.t())
        A_1d = D2_inv - torch.sparse.mm(torch.sparse.mm(B1.t(), D1_inv), B1)
        A_1u_norm = torch.sparse.mm((sparse_diag_identity(A_1u.shape[0]) + A_1u),
                                    (torch.sparse_coo_tensor(D2_indices, 1 / (D2diag + 1), (x1, x1))))
        A_1d_norm = torch.sparse.mm((D2 + sparse_diag_identity(D2.shape[0])),
                                    (A_1d + sparse_diag_identity(A_1d.shape[0])))
        L1 = A_1u_norm + A_1d_norm

        B2_sum = D2diag_1
        B2_sum_inv = 1 / (B2_sum + 1)
        D5inv = sparse_diag(B2_sum_inv)
        A_2d = sparse_diag_identity(B2.shape[1]) + torch.sparse.mm(torch.sparse.mm(B2.t(), D5inv), B2)
        A_2d_norm = torch.sparse.mm((2 * sparse_diag_identity(B2.shape[1])),
                                    (A_2d + sparse_diag_identity(A_2d.shape[0])))
        L2 = A_2d_norm

        B2D3 = torch.sparse.mm(B2, D3)
        D2B1TD1inv = (1 / np.sqrt(2.)) * torch.sparse.mm(torch.sparse.mm(D2, B1.t()), D1_inv)
        D1invB1 = (1 / np.sqrt(2.)) * torch.sparse.mm(D1_inv, B1)
        B2TD2inv = torch.sparse.mm(B2.t(), D5inv)

        return SimplicialObject(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv, label)

    def collate(self, data_list):
        X0, X1, X2 = [], [], []
        L0, L1, L2 = [], [], []
        # D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
        D1, D2, D3, D4 = [], [], [], []
        label = []

        x0_total, x1_total, x2_total = 0, 0, 0
        l0_total, l1_total, l2_total = 0, 0, 0
        d1_total, d2_total, d3_total, d4_total = 0, 0, 0, 0
        label_total = 0

        slices = {"X0": [0],
                  "X1": [0],
                  "X2": [0],
                  "L0": [0],
                  "L1": [0],
                  "L2": [0],
                  "B2D3": [0],
                  "D2B1TD1inv": [0],
                  "D1invB1": [0],
                  "B2TD2inv": [0],
                  "label": [0]}

        for data in data_list:
            x0, x1, x2 = data.X0, data.X1, data.X2
            l0, l1, l2 = data.L0, data.L1, data.L2
            d1, d2, d3, d4 = data.B2D3, data.D2B1TD1inv, data.D1invB1, data.B2TD2inv

            l = data.label.shape[0]
            x0_s, x1_s, x2_s = x0.shape[0], x1.shape[0], x2.shape[0]
            l0_s, l1_s, l2_s = l0.shape[1], l1.shape[1], l2.shape[1]
            d1_s, d2_s, d3_s, d4_s = d1.shape[1], d2.shape[1], d3.shape[1], d4.shape[1]

            X0.append(x0)
            X1.append(x1)
            X2.append(x2)
            L0.append(l0)
            L1.append(l1)
            L2.append(l2)
            D1.append(d1)
            D2.append(d2)
            D3.append(d3)
            D4.append(d4)

            label.append(data.label)

            x0_total += x0_s
            x1_total += x1_s
            x2_total += x2_s
            l0_total += l0_s
            l1_total += l1_s
            l2_total += l2_s
            d1_total += d1_s
            d2_total += d2_s
            d3_total += d3_s
            d4_total += d4_s
            label_total += l

            slices["X0"].append(x0_total)
            slices["X1"].append(x1_total)
            slices["X2"].append(x2_total)
            slices["L0"].append(l0_total)
            slices["L1"].append(l1_total)
            slices["L2"].append(l2_total)
            slices["B2D3"].append(d1_total)
            slices["D2B1TD1inv"].append(d2_total)
            slices["D1invB1"].append(d3_total)
            slices["B2TD2inv"].append(d4_total)
            slices["label"].append(label_total)

        X0 = torch.cat(X0, dim=0).cpu()
        X1 = torch.cat(X1, dim=0).cpu()
        X2 = torch.cat(X2, dim=0).cpu()
        L0 = torch.cat(L0, dim=-1).cpu()
        L1 = torch.cat(L1, dim=-1).cpu()
        L2 = torch.cat(L2, dim=-1).cpu()
        D1 = torch.cat(D1, dim=-1).cpu()
        D2 = torch.cat(D2, dim=-1).cpu()
        D3 = torch.cat(D3, dim=-1).cpu()
        D4 = torch.cat(D4, dim=-1).cpu()
        label = torch.cat(label, dim=-1).cpu()
        # D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
        data = SimplicialObject(X0, X1, X2, L0, L1, L2, D1, D2, D3, D4, label)

        return data, slices

    def get(self, data, slices, idx):
        x0_slice = slices["X0"][idx:idx + 2]
        x1_slice = slices["X1"][idx:idx + 2]
        x2_slice = slices["X2"][idx:idx + 2]
        l0_slice = slices["L0"][idx:idx + 2]
        l1_slice = slices["L1"][idx:idx + 2]
        l2_slice = slices["L2"][idx:idx + 2]
        d1_slice = slices["B2D3"][idx:idx + 2]
        d2_slice = slices["D2B1TD1inv"][idx:idx + 2]
        d3_slice = slices["D1invB1"][idx:idx + 2]
        d4_slice = slices["B2TD2inv"][idx:idx + 2]
        label_slice = slices["label"][idx: idx + 2]

        X0 = data.X0[x0_slice[0]: x0_slice[1]]
        X1 = data.X1[x1_slice[0]: x1_slice[1]]
        X2 = data.X2[x2_slice[0]: x2_slice[1]]

        L0 = data.L0[:, l0_slice[0]: l0_slice[1]]
        L1 = data.L1[:, l1_slice[0]: l1_slice[1]]
        L2 = data.L2[:, l2_slice[0]: l2_slice[1]]

        # D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
        D1 = data.B2D3[:, d1_slice[0]: d1_slice[1]]
        D2 = data.D2B1TD1inv[:, d2_slice[0]: d2_slice[1]]
        D3 = data.D1invB1[:, d3_slice[0]: d3_slice[1]]
        D4 = data.B2TD2inv[:, d4_slice[0]: d4_slice[1]]

        label = data.label[label_slice[0]: label_slice[1]]

        return SimplicialObject(X0, X1, X2, L0, L1, L2, D1, D2, D3, D4, label)

    def batch(self, objectList):
        def unpack_SimplicialObject(SimplicialObject):
            X0, X1, X2 = SimplicialObject.X0, SimplicialObject.X1, SimplicialObject.X2
            L0, L1, L2 = SimplicialObject.L0, SimplicialObject.L1, SimplicialObject.L2

            L0_i, L0_v = L0[0:2], L0[2:3].squeeze()
            L1_i, L1_v = L1[0:2], L1[2:3].squeeze()
            L2_i, L2_v = L2[0:2], L2[2:3].squeeze()

            D1, D2, D3, D4 = SimplicialObject.B2D3, SimplicialObject.D2B1TD1inv, \
                             SimplicialObject.D1invB1, SimplicialObject.B2TD2inv

            D1_i, D1_v = D1[0:2], D1[2:3].squeeze()
            D2_i, D2_v = D2[0:2], D2[2:3].squeeze()
            D3_i, D3_v = D3[0:2], D3[2:3].squeeze()
            D4_i, D4_v = D4[0:2], D4[2:3].squeeze()
            # D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
            label = SimplicialObject.label
            return [X0, X1, X2], [L0_i, L1_i, L2_i], [L0_v, L1_v, L2_v], \
                   [D1_i, D2_i, D3_i, D4_i], [D1_v, D2_v, D3_v, D4_v], label

        unpacked_SimplicialObject = [unpack_SimplicialObject(g) for g in objectList]
        X, L_i, L_v, D_i, D_v, labels = [*zip(*unpacked_SimplicialObject)]
        X, L_i, L_v, D_i, D_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)], [*zip(*D_i)], [*zip(*D_v)]
        features_dct = batch_all_feature_and_lapacian_pair(X, L_i, L_v)
        D_i_batch, D_v_batch = [], []
        for i, v in zip(D_i, D_v):
            sizes_x = [1 + int(torch.max(matrix[0]).item()) for matrix in i]
            sizes_y = [1 + int(torch.max(matrix[1]).item()) for matrix in i]
            d_i_batch, d_v_batch = batch_sparse_matrix(i, v, sizes_x, sizes_y)
            D_i_batch.append(d_i_batch)
            D_v_batch.append(d_v_batch)
        features_dct['d_indices'] = D_i_batch
        features_dct['d_values'] = D_v_batch

        labels = torch.cat(labels, dim=0)
        return features_dct, labels

    def clean_feature_dct(self, feature_dct):
        feature_dct = convert_indices_and_values_to_sparse(feature_dct, 'lapacian_indices', 'lapacian_values',
                                                           'lapacian')
        return convert_indices_and_values_to_sparse(feature_dct, 'd_indices', 'd_values', 'others')

    def repair(self, feature_dct):
        # The last few rows/columns of the matrix might be empty and this info is lost when you convert it to a sparse matrix
        _, X, _ = unpack_feature_dct_to_L_X_B(feature_dct)

        X0, X1, X2 = X

        n, e, t = X0.shape[0], X1.shape[0], X2.shape[0]

        ideal_shape = [(e, t), (e, n), (n, e), (t, e)]
        feature_dct['others'] = [repair_sparse(m, s) for m, s in zip(feature_dct['others'], ideal_shape)]

        return feature_dct
