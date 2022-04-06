import torch
from models.ProcessorTemplate import NNProcessor
from utils import ensure_input_is_tensor
from models.nn_utils import to_sparse_coo
from models.nn_utils import batch_all_feature_and_lapacian_pair, normalise, batch_sparse_matrix
from models.SimplicialComplex import SimplicialComplex
from constants import DEVICE


class SANComplex(SimplicialComplex):

    def __init__(self, X0, X1, X2, L0, L1, L1_up, L1_down, L2, label, batch=None):
        super().__init__(X0, X1, X2, L0, L1, L2, label, batch=batch)

        self.L1_up = ensure_input_is_tensor(L1_up)
        self.L1_down = ensure_input_is_tensor(L1_down)

    def __eq__(self, other):
        x0 = torch.allclose(self.X0, other.X0, atol=1e-5)
        x1 = torch.allclose(self.X1, other.X1, atol=1e-5)
        x2 = torch.allclose(self.X2, other.X2, atol=1e-5)
        l0 = torch.allclose(self.L0, other.L0, atol=1e-5)
        l1 = torch.allclose(self.L1, other.L1, atol=1e-5)
        l1_u = torch.allclose(self.L1_up, other.L1_up, atol=1e-5)
        l1_d = torch.allclose(self.L1_down, other.L1_down, atol=1e-5)
        l2 = torch.allclose(self.L2, other.L2, atol=1e-5)
        label = torch.allclose(self.label, other.label, atol=1e-5)
        return all([x0, x1, x2, l0, l1, l1_u, l1_d, l2, label])

    def unpack_up_down(self):
        return [self.L1_up, self.L1_down]
    
    def to_device(self):
        super().to_device()
        self.L1_up = self.L1_up.to(DEVICE)
        self.L1_down = self.L1_down.to(DEVICE)


class SANProcessor(NNProcessor):

    def process(self, CoChain):
        b1, b2 = to_sparse_coo(CoChain.b1).cpu(), to_sparse_coo(CoChain.b2).cpu()

        X0, X1, X2 = CoChain.X0, CoChain.X1, CoChain.X2

        L0 = normalise(torch.sparse.mm(b1, b1.t()))
        L1_up = torch.sparse.mm(b2, b2.t())
        L1_down = torch.sparse.mm(b1.t(), b1)
        L1 = normalise(L1_up + L1_down)
        L2 = normalise(torch.sparse.mm(b2.t(), b2))

        assert (X0.shape[0] == L0.shape[0])
        assert (X1.shape[0] == L1_up.shape[0])
        assert (X1.shape[0] == L1_down.shape[0])
        assert (X2.shape[0] == L2.shape[0])

        label = CoChain.label

        return SANComplex(X0, X1, X2, L0, L1, L1_up, L1_down, L2, label)

    def collate(self, data_list):
        X0, X1, X2 = [], [], []
        L0, L1_up, L1_dn, L1, L2 = [], [], [], [], []
        label = []

        x0_total, x1_total, x2_total = 0, 0, 0
        l0_total, l1_u_total, l1_d_total, l1_total, l2_total = 0, 0, 0, 0, 0
        label_total = 0

        slices = {"X0": [0],
                  "X1": [0],
                  "X2": [0],
                  "L0": [0],
                  "L1_up": [0],
                  "L1_down": [0],
                  "L1": [0],
                  "L2": [0],
                  "label": [0]}

        for data in data_list:
            x0, x1, x2 = data.X0, data.X1, data.X2
            l0, l1_up, l1_dn, l1, l2 = data.L0, data.L1_up, data.L1_down, data.L1, data.L2
            l = data.label

            x0_s, x1_s, x2_s = x0.shape[0], x1.shape[0], x2.shape[0]
            l0_s, l1_u_s, l1_d_s, l1_s, l2_s = l0.shape[1], l1_up.shape[1], l1_dn.shape[1], l1.shape[1], l2.shape[1]
            l_s = l.shape[0]

            X0.append(x0)
            X1.append(x1)
            X2.append(x2)
            L0.append(l0)
            L1_up.append(l1_up)
            L1_dn.append(l1_dn)
            L1.append(l1)
            L2.append(l2)
            label.append(l)

            x0_total += x0_s
            x1_total += x1_s
            x2_total += x2_s
            l0_total += l0_s
            l1_u_total += l1_u_s
            l1_d_total += l1_d_s
            l1_total += l1_s
            l2_total += l2_s
            label_total += l_s

            slices["X0"].append(x0_total)
            slices["X1"].append(x1_total)
            slices["X2"].append(x2_total)
            slices["L0"].append(l0_total)
            slices["L1_up"].append(l1_u_total)
            slices["L1_down"].append(l1_d_total)
            slices["L1"].append(l1_total)
            slices["L2"].append(l2_total)
            slices["label"].append(label_total)

            del data

        del data_list

        X0 = torch.cat(X0, dim=0).cpu()
        X1 = torch.cat(X1, dim=0).cpu()
        X2 = torch.cat(X2, dim=0).cpu()
        L0 = torch.cat(L0, dim=-1).cpu()
        L1_up = torch.cat(L1_up, dim=-1).cpu()
        L1_down = torch.cat(L1_dn, dim=-1).cpu()
        L1 = torch.cat(L1, dim=-1).cpu()
        L2 = torch.cat(L2, dim=-1).cpu()
        label = torch.cat(label, dim=-1).cpu()

        data = SANComplex(X0, X1, X2, L0, L1, L1_up, L1_down, L2, label)

        return data, slices

    def get(self, data, slices, idx):
        x0_slice = slices["X0"][idx:idx + 2]
        x1_slice = slices["X1"][idx:idx + 2]
        x2_slice = slices["X2"][idx:idx + 2]
        l0_slice = slices["L0"][idx:idx + 2]
        l1_u_slice = slices["L1_up"][idx:idx + 2]
        l1_d_slice = slices["L1_down"][idx:idx + 2]
        l1_slice = slices["L1"][idx:idx + 2]
        l2_slice = slices["L2"][idx:idx + 2]
        label_slice = slices["label"][idx: idx + 2]

        X0 = data.X0[x0_slice[0]: x0_slice[1]]
        X1 = data.X1[x1_slice[0]: x1_slice[1]]
        X2 = data.X2[x2_slice[0]: x2_slice[1]]

        L0 = data.L0[:, l0_slice[0]: l0_slice[1]]
        L1_up = data.L1_up[:, l1_u_slice[0]: l1_u_slice[1]]
        L1_dn = data.L1_down[:, l1_d_slice[0]: l1_d_slice[1]]
        L1 = data.L1[:, l1_slice[0]: l1_slice[1]]
        L2 = data.L2[:, l2_slice[0]: l2_slice[1]]

        label = data.label[label_slice[0]: label_slice[1]]

        return SANComplex(X0, X1, X2, L0, L1, L1_up, L1_dn, L2, label)

    def batch(self, objectList):
        def unpack_SimplicialObject(SimplicialObject):
            X0, X1, X2 = SimplicialObject.X0, SimplicialObject.X1, SimplicialObject.X2
            L0, L1_u, L1_d, L1, L2 = SimplicialObject.L0, SimplicialObject.L1_up, SimplicialObject.L1_down, SimplicialObject.L1, SimplicialObject.L2

            L0_i, L0_v = L0[0:2], L0[2:3].squeeze()
            L1_u_i, L1_u_v = L1_u[0:2], L1_u[2:3].squeeze()
            L1_d_i, L1_d_v = L1_d[0:2], L1_d[2:3].squeeze()
            L1_i, L1_v = L1[0:2], L1[2:3].squeeze()
            L2_i, L2_v = L2[0:2], L2[2:3].squeeze()

            label = SimplicialObject.label
            return [X0, X1, X2], [L0_i, L1_i, L2_i], [L0_v, L1_v, L2_v], \
                   [L1_u_i, L1_d_i], [L1_u_v, L1_d_v], label

        unpacked_grapObject = [unpack_SimplicialObject(g) for g in objectList]
        X, L_i, L_v, D_i, D_v, labels = [*zip(*unpacked_grapObject)]
        X, L_i, L_v, D_i, D_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)], [*zip(*D_i)], [*zip(*D_v)]

        features_dct = batch_all_feature_and_lapacian_pair(X, L_i, L_v)
        D_i_batch, D_v_batch = [], []
        for i, v in zip(D_i, D_v):
            sizes_x = [matrix.shape[0] for matrix in X[1]]
            sizes_y = [matrix.shape[0] for matrix in X[1]]
            d_i_batch, d_v_batch = batch_sparse_matrix(i, v, sizes_x, sizes_y)
            D_i_batch.append(d_i_batch)
            D_v_batch.append(d_v_batch)
        features_dct['d_indices'] = D_i_batch
        features_dct['d_values'] = D_v_batch

        labels = torch.cat(labels, dim=0)
        X0, X1, X2 = features_dct['features']
        L0_i, L1_i, L2_i = features_dct['lapacian_indices']
        L0_v, L1_v, L2_v = features_dct['lapacian_values']
        L1_u_i, L1_d_i = features_dct['d_indices']
        L1_u_v, L1_d_v= features_dct['d_values']

        L0 = torch.cat([L0_i, L0_v.unsqueeze(0)], dim=0)
        L1 = torch.cat([L1_i, L1_v.unsqueeze(0)], dim=0)
        L2 = torch.cat([L2_i, L2_v.unsqueeze(0)], dim=0)

        L1_u = torch.cat([L1_u_i, L1_u_v.unsqueeze(0)], dim=0)
        L1_d = torch.cat([L1_d_i, L1_d_v.unsqueeze(0)], dim=0)

        batch = features_dct['batch_index']
        
        complex = SANComplex(X0, X1, X2, L0, L1, L1_u, L1_d, L2, torch.tensor([0]), batch)
        return complex, labels

    def clean_features(self, sanComplex):
        sanComplex.L0 = to_sparse_coo(sanComplex.L0)
        sanComplex.L1 = to_sparse_coo(sanComplex.L1)
        sanComplex.L1_up = to_sparse_coo(sanComplex.L1_up)
        sanComplex.L1_down = to_sparse_coo(sanComplex.L1_down)
        sanComplex.L2 = to_sparse_coo(sanComplex.L2)
        return sanComplex

    def repair(self, sanComplex):
        return sanComplex

