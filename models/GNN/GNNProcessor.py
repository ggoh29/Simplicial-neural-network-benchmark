from utils import sparse_to_tensor, ensure_input_is_tensor
import torch
from models.nn_utils import to_sparse_coo
from models.ProcessorTemplate import NNProcessor
from models.nn_utils import batch_feature_and_lapacian_pair, convert_indices_and_values_to_sparse, \
    scipy_sparse_to_torch_sparse
import scipy.sparse as sp
import numpy as np


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GraphObject:

    def __init__(self, X0, L0, label):
        self.X0 = X0

        self.L0 = ensure_input_is_tensor(L0)

        self.label = label

    def __eq__(self, other):
        x0 = torch.allclose(self.X0, other.X0, atol=1e-5)
        L0 = torch.allclose(self.L0, other.L0, atol=1e-5)
        l0 = torch.allclose(self.label, other.label, atol=1e-5)
        return all([x0, L0, l0])


class GNNProcessor(NNProcessor):

    def process(self, scData):
        b1, b2 = to_sparse_coo(scData.b1), to_sparse_coo(scData.b2)

        X0 = scData.X0
        L0 = torch.sparse.mm(b1, b1.t()).to_dense()
        L0[L0 != 0] = 1
        L0 = normalize_adj(L0.cpu().numpy())
        L0 = scipy_sparse_to_torch_sparse(L0)
        label = scData.label

        return GraphObject(X0, L0, label)

    def collate(self, data_list):
        X0 = []
        L0 = []
        label = []

        X0_total = 0
        L0_total = 0
        label_total = 0

        slices = {"X0": [0], "L0": [0], "label": [0]}

        for data in data_list:
            x0 = data.X0
            l0 = data.L0
            l = data.label

            x0_s = x0.shape[0]
            l0_s = l0.shape[1]
            label_s = l.shape[0]

            X0.append(x0)
            L0.append(l0)
            label.append(l)

            X0_total += x0_s
            L0_total += l0_s
            label_total += label_s

            slices["X0"].append(X0_total)
            slices["L0"].append(L0_total)
            slices["label"].append(label_total)

        X0 = torch.cat(X0, dim=0).to('cpu')
        L0 = torch.cat(L0, dim=-1).to('cpu')
        label = torch.cat(label, dim=-1).to('cpu')

        data = GraphObject(X0, L0, label)

        return data, slices

    def get(self, data, slices, idx):
        x0_slice = slices["X0"][idx:idx + 2]
        l0_slice = slices["L0"][idx:idx + 2]
        label_slice = slices["label"][idx: idx + 2]

        X0 = data.X0[x0_slice[0]: x0_slice[1]]
        L0 = data.L0[:, l0_slice[0]: l0_slice[1]]
        label = data.label[label_slice[0]: label_slice[1]]
        return GraphObject(X0, L0, label)

    def batch(self, objectList):
        def unpack_graphObject(graphObject):
            features = graphObject.X0
            lapacian = graphObject.L0
            indices = lapacian[0:2]
            values = lapacian[2:3].squeeze()
            label = graphObject.label
            return features, indices, values, label

        unpacked_grapObject = [unpack_graphObject(g) for g in objectList]
        X, L_i, L_v, labels = [*zip(*unpacked_grapObject)]
        X, L_i, L_v = list(X), list(L_i), list(L_v)
        X_batch, I_batch, V_batch, batch_index = batch_feature_and_lapacian_pair(X, L_i, L_v)
        features_dct = {'features': [X_batch],
                        'lapacian_indices': [I_batch],
                        'lapacian_values': [V_batch],
                        'batch_index': [batch_index]}

        labels = torch.cat(labels, dim=0)
        return features_dct, labels

    def clean_feature_dct(self, feature_dct):
        return convert_indices_and_values_to_sparse(feature_dct, 'lapacian_indices', 'lapacian_values', 'lapacian')

    def repair(self, feature_dct):
        return feature_dct
