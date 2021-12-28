from models.nn_utils import unpack_feature_dct_to_L_X_B, convert_to_SC, torch_sparse_to_scipy_sparse, repair_sparse,\
    scipy_sparse_to_torch_sparse, to_sparse_coo
import scipy
import numpy as np
import torch
import torch.nn as nn
from constants import DEVICE

def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]

def corruption_function(feature_dct, processor_type, p = 0):
    L, X, batch = unpack_feature_dct_to_L_X_B(feature_dct)
    X0 = X[0]
    nb_nodes = X0.shape[0]
    idx = np.random.permutation(nb_nodes)
    C_X0 = X0[idx]

    L0_i = L[0].coalesce().indices().to(DEVICE)
    L0_v = -torch.ones(L0_i.shape[1]).to(DEVICE)
    L0 = torch.sparse_coo_tensor(L0_i, L0_v).to(DEVICE)
    cor_adj_i = torch.triu_indices(nb_nodes, nb_nodes, 0).to(DEVICE)
    cor_adj_v = torch.tensor(np.random.binomial(1, p, size=(cor_adj_i.shape[1])), dtype=torch.float, device = DEVICE)

    # logical xor for edge insertion/deletion
    cor_adj = torch.sparse_coo_tensor(cor_adj_i, cor_adj_v).to(DEVICE)
    cor_adj = L0 + cor_adj
    cor_adj_i, cor_adj_v = cor_adj.coalesce().indices().to(DEVICE), cor_adj.coalesce().values().to(DEVICE)
    cor_adj_v = torch.abs(cor_adj_v)
    cor_adj = torch.sparse_coo_tensor(cor_adj_i, cor_adj_v)
    cor_adj = torch_sparse_to_scipy_sparse(cor_adj)
    cor_adj = scipy.sparse.triu(cor_adj, k=1)
    cor_adj.eliminate_zeros()
    cor_adj = scipy_sparse_to_torch_sparse(cor_adj)

    fake_labels = torch.tensor([0 for _ in range(nb_nodes)], dtype=torch.float, device = DEVICE)
    scData = convert_to_SC(cor_adj, C_X0, fake_labels)
    corrupted_train = processor_type.process(scData)
    corrupted_train = processor_type.batch([corrupted_train])[0]
    corrupted_train = processor_type.clean_feature_dct(corrupted_train)
    corrupted_train = processor_type.repair(corrupted_train)

    return corrupted_train

######################################################################################################
# This section is adopted from https://github.com/PetarV-/DGI/tree/61baf67d7052905c77bdeb28c22926f04e182362
######################################################################################################

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m.to(DEVICE))

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class DGI(nn.Module):
    def __init__(self, input_size, output_size, model):
        super(DGI, self).__init__()
        self.model = model(input_size, output_size).to(DEVICE)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(output_size)

    def forward(self, feature_dct, processor_type):
        corrupted_dct = corruption_function(feature_dct, processor_type)

        feature_dct = {key: convert_to_device(feature_dct[key]) for key in feature_dct}
        h_1 = self.model(feature_dct).unsqueeze(0)
        c = self.read(h_1, None)
        c = self.sigm(c)

        corrupted_dct = {key: convert_to_device(corrupted_dct[key]) for key in corrupted_dct}
        h_2 = self.model(corrupted_dct).unsqueeze(0)

        ret = self.disc(c, h_1, h_2)

        return ret

    # Detach the return variables
    def embed(self, feature_dct):
        feature_dct = {key: convert_to_device(feature_dct[key]) for key in feature_dct}
        h_1 = self.model(feature_dct)
        c = self.read(h_1, None)

        return h_1.detach(), c.detach()