from utils import ensure_input_is_tensor
import torch
from models.nn_utils import to_sparse_coo
from models.ProcessorTemplate import NNProcessor
from models.nn_utils import batch_feature_and_lapacian_pair, repair_sparse
from models.SimplicialComplex import SimplicialComplex
from constants import DEVICE


class GraphComplex(SimplicialComplex):

    def __init__(self, X0, L0, label, batch=None):
        super().__init__(X0, None, None, L0, None, None, label, batch=batch)

    def __eq__(self, other):
        x0 = torch.allclose(self.X0, other.X0, atol=1e-5)
        L0 = torch.allclose(self.L0, other.L0, atol=1e-5)
        l0 = torch.allclose(self.label, other.label, atol=1e-5)
        return all([x0, L0, l0])

    def to_device(self):
        self.X0 = self.X0.to(DEVICE)
        self.L0 = self.L0.to(DEVICE)
        self.batch = [batch.to(DEVICE) for batch in self.batch]


class GNNProcessor(NNProcessor):

    def process(self, CoChain):
        b1 = to_sparse_coo(CoChain.b1)

        X0 = CoChain.X0
        L0 = torch.sparse.mm(b1, b1.t())
        label = CoChain.label

        return GraphComplex(X0, L0, label)

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

        X0 = torch.cat(X0, dim=0).cpu()
        L0 = torch.cat(L0, dim=-1).cpu()
        label = torch.cat(label, dim=-1).cpu()

        data = GraphComplex(X0, L0, label)

        return data, slices

    def get(self, data, slices, idx):
        x0_slice = slices["X0"][idx:idx + 2]
        l0_slice = slices["L0"][idx:idx + 2]
        label_slice = slices["label"][idx: idx + 2]

        X0 = data.X0[x0_slice[0]: x0_slice[1]]
        L0 = data.L0[:, l0_slice[0]: l0_slice[1]]
        label = data.label[label_slice[0]: label_slice[1]]
        return GraphComplex(X0, L0, label)

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
        X0 = features_dct['features'][0]
        L0_i = features_dct['lapacian_indices'][0]
        L0_v = features_dct['lapacian_values'][0]

        L0 = torch.cat([L0_i, L0_v.unsqueeze(0)], dim=0)

        batch = features_dct['batch_index']

        complex = GraphComplex(X0, L0, torch.tensor([0]), batch)

        return complex, labels

    def clean_features(self, simplicialComplex):
        simplicialComplex.L0 = to_sparse_coo(simplicialComplex.L0)
        return simplicialComplex

    def repair(self, simplicialComplex):
        X0 = simplicialComplex.X0
        n = X0.shape[0]
        L0 = simplicialComplex.L0

        simplicialComplex.L0 = repair_sparse(L0, (n, n))

        return simplicialComplex
