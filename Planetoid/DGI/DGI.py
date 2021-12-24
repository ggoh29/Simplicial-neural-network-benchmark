from models.nn_utils import unpack_feature_dct_to_L_X_B
from Planetoid.PlanetoidDataset.PlanetoidLoader import convert_to_SC
import numpy as np
import torch
import torch.nn as nn
from constants import DEVICE

def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]

def corruption_function(feature_dct, processor_type, p = 0.000):
    L, X, batch = unpack_feature_dct_to_L_X_B(feature_dct)
    X0 = X[0]

    nb_nodes = X0.shape[0]
    idx = np.random.permutation(nb_nodes)
    C_X0 = X0[idx]

    L0_i = L[0].coalesce().indices()
    L0_v = torch.ones(L0_i.shape[1])
    L0 = torch.sparse_coo_tensor(L0_i, L0_v).to_dense()

    C_L0 = torch.tensor(np.random.binomial(1, p, size = (L[0].shape)), dtype = torch.float)
    C_L0 = torch.logical_xor(L0, C_L0)
    C_L0 = torch.triu(C_L0, diagonal=1).to_sparse()
    C_L0_i = C_L0.coalesce().indices()
    ones = torch.ones(C_L0_i.shape[1])

    fake_labels = torch.tensor([0 for _ in range(nb_nodes)])
    scData = convert_to_SC(ones, C_L0_i, C_X0, fake_labels, False)
    corrupted_train = processor_type.process(scData)
    corrupted_train = processor_type.batch([corrupted_train])[0]
    corrupted_train = processor_type.clean_feature_dct(corrupted_train)
    corrupted_train = processor_type.repair(corrupted_train)

    # Might be missing nodes from adjacency matrix, add padding here
    L0_diag = torch.diag(L0, 0)
    L0_diag = torch.diag(L0_diag)
    adj = corrupted_train['lapacian'][0].to_dense()
    L0_diag[: adj.shape[0], : adj.shape[1]] = adj
    corrupted_train['lapacian'][0] = L0_diag.to_sparse()
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

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1).unsqueeze(0)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1).unsqueeze(0)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

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
        h_1 = self.model(feature_dct)
        c = self.read(h_1, None)
        c = self.sigm(c)

        corrupted_dct = {key: convert_to_device(corrupted_dct[key]) for key in corrupted_dct}
        h_2 = self.model(corrupted_dct)
        ret = self.disc(c, h_1, h_2, None, None)

        return ret

    # Detach the return variables
    def embed(self, feature_dct):
        h_1 = self.model(feature_dct)
        c = self.read(h_1, None)

        return h_1.detach(), c.detach()