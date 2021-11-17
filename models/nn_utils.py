import torch
import scipy.sparse.linalg as spl
import numpy as np
from scipy.sparse import coo_matrix


def to_sparse_coo(matrix):
    indices = matrix[0:2]
    values = matrix[2:3].squeeze()
    return torch.sparse_coo_tensor(indices, values)


def chebyshev(L, X, k=3):
    dp = [X, torch.sparse.mm(L, X)]
    for i in range(2, k):
        nxt = 2*(torch.sparse.mm(L, dp[i-1]))
        dp.append(torch.sparse.FloatTensor.add(nxt, -(dp[i-2])))
    return torch.cat(dp, dim=1)

def torch_sparse_to_scipy_sparse(matrix):
    i = matrix.coalesce().indices()
    v = matrix.coalesce().values()

    (m, n) = matrix.shape[0], matrix.shape[1]
    return coo_matrix((v, i), shape=(m, n))


def scipy_sparse_to_torch_sparse(matrix):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v)


def normalise(L):

    M = L.shape[0]
    L = torch_sparse_to_scipy_sparse(L)

    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    ret = L.copy()
    ret *= 2.0 / topeig
    ret.setdiag(ret.diagonal(0) - np.ones(M), 0)

    return scipy_sparse_to_torch_sparse(ret)


def scData_to_simplicial0(scData):

    b1, b2 = to_sparse_coo(scData.b1), to_sparse_coo(scData.b2)

    X0, X1, X2 = scData.X0, scData.X1, scData.X2
    L0 = torch.sparse.mm(b1, b1.t())

    # splitting the sparse tensor as pooling cannot return sparse and to make preparation for minibatching easier
    assert (X0.size()[0] == L0.size()[0])
    return [[X0], [L0.coalesce().indices()], [L0.coalesce().values()]], scData.label


def scData_to_simplicial1(scData):
    b1, b2 = to_sparse_coo(scData.b1), to_sparse_coo(scData.b2)

    X0, X1, X2 = scData.X0, scData.X1, scData.X2
    L0 = torch.sparse.mm(b1, b1.t())

    L1 = torch.sparse.mm(b1.t(), b1)

    # splitting the sparse tensor as pooling cannot return sparse and to make preparation for minibatching easier
    assert (X0.size()[0] == L0.size()[0])
    assert (X1.size()[0] == L1.size()[0])
    return [[X0, X1], [L0.coalesce().indices(), L1.coalesce().indices()],
            [L0.coalesce().values(), L1.coalesce().values()]], scData.label


def scData_to_simplicial2(scData):

    b1, b2 = to_sparse_coo(scData.b1), to_sparse_coo(scData.b2)

    X0, X1, X2 = scData.X0, scData.X1, scData.X2
    L0 = torch.sparse.mm(b1, b1.t())

    L1 = torch.sparse.FloatTensor.add(torch.sparse.mm(b1.t(), b1), torch.sparse.mm(b2, b2.t()))
    L2 = torch.sparse.mm(b2.t(), b2)

    # splitting the sparse tensor as pooling cannot return sparse and to make preparation for minibatching easier
    assert (X0.size()[0] == L0.size()[0])
    assert (X1.size()[0] == L1.size()[0])
    assert (X2.size()[0] == L2.size()[0])

    return [[X0, X1, X2],
            [L0.coalesce().indices(), L1.coalesce().indices(), L2.coalesce().indices()],
            [L0.coalesce().values(), L1.coalesce().values(), L2.coalesce().values()]], scData.label


def collated_data_to_batch(samples, fn = scData_to_simplicial0):
    LapacianData = [fn(scData) for scData in samples]
    Lapacians, labels = map(list, zip(*LapacianData))
    labels = torch.cat(labels, dim=0)
    feature_dct = _sanitize_input_for_batch(Lapacians)

    return feature_dct, labels


def _sanitize_input_for_batch(input_list):

    X, L_i, L_v = [*zip(*input_list)]
    X, L_i, L_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)]

    return batch_all_feature_and_lapacian_pair(X, L_i, L_v)


def batch_all_feature_and_lapacian_pair(X, L_i, L_v):

    X_batch, I_batch, V_batch, batch_index = [], [], [], []
    for i in range(len(X)):
        x, i, v, batch = batch_feature_and_lapacian_pair(X[i], L_i[i], L_v[i])
        X_batch.append(x)
        I_batch.append(i)
        V_batch.append(v)
        batch_index.append(batch)

    features_dct = {'features' : X_batch,
                    'lapacian_indices' : I_batch,
                    'lapacian_values' : V_batch,
                    'batch_index' : batch_index}

    # I_batch and V_batch form the indices and values of coo_sparse tensor but sparse tensors
    # cant be stored so storing them as two separate tensors
    return features_dct


def batch_feature_and_lapacian_pair(x_list, L_i_list, L_v_list):
    feature_batch = torch.cat(x_list, dim=0)
    sizes = [*map(lambda x: x.size()[0], x_list)]

    I_cat, V_cat = _batch_sparse_matrix(L_i_list, L_v_list, sizes)
    # lapacian_batch = torch.sparse_coo_tensor(L_cat, V_cat)
    batch = [[i for _ in range(sizes[i])] for i in range(len(sizes))]
    batch = torch.tensor([i for sublist in batch for i in sublist])
    return feature_batch, I_cat, V_cat, batch


def _batch_sparse_matrix(L_i_list, L_v_list, sizes):
    L_i_list = list(L_i_list)
    mx = 0
    for i in range(1, len(L_i_list)):
        mx += sizes[i - 1]
        L_i_list[i] += mx
    I_cat = torch.cat(L_i_list, dim=1)
    V_cat = torch.cat(L_v_list, dim=0)
    return I_cat, V_cat

def convert_indices_and_values_to_sparse(feature_dct, indices_key, value_key, output_key):
    lapacian = []
    indices, values = feature_dct[indices_key], feature_dct[value_key]
    for i, v in zip(indices, values):
        lapacian.append(torch.sparse_coo_tensor(i, v))
    feature_dct[output_key] = lapacian
    feature_dct.pop(indices_key)
    feature_dct.pop(value_key)
    return feature_dct


def unpack_feature_dct_to_L_X_B(dct):
    # unpack to lapacian, features and batch
    return dct['lapacian'], dct['features'], dct['batch_index']



