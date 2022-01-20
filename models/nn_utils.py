import torch
import scipy
import scipy.sparse.linalg as spl
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from utils import edge_to_node_matrix, triangle_to_edge_matrix
from models.SCData import SCData
import functools
import scipy.sparse as sp


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features, dtype=torch.float)


def remove_diag_sparse(sparse_adj):
    scipy_adj = torch_sparse_to_scipy_sparse(sparse_adj)
    scipy_adj = scipy.sparse.triu(scipy_adj, k=1)
    return scipy_sparse_to_torch_sparse(scipy_adj)



def get_features(features, sc_list):
    def _get_features(features, sc):
        f = [features[i] for i in sc]
        return functools.reduce(lambda a, b: a + b, f).float()
        # return functools.reduce(lambda a, b: torch.logical_and(a, b), f).float()
        # return a/torch.sum(s)
    features = [_get_features(features, sc) for sc in sc_list]
    if bool(features):
        return torch.stack(features, dim = 0)
    else:
        return torch.tensor([])

def filter_simplices(node_features, simplice):
    s = [node_features[i] for i in simplice]
    common_features = functools.reduce(lambda a, b: torch.logical_and(a, b), s).float()
    return torch.sum(common_features).item() > 0


def convert_to_SC(adj, features, labels, X1 = None, X2 = None):
    X0 = features

    nodes = [i for i in range(X0.shape[0])]
    edges = adj.coalesce().indices().tolist()
    edges = [(i, j) for i, j in zip(edges[0], edges[1])]
    # edges = [*filter(lambda x: filter_simplices(features, x), edges)]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    triangles = [list(sorted(x)) for x in nx.enumerate_all_cliques(g) if len(x) == 3]
    # triangles = [*filter(lambda x: filter_simplices(features, x), triangles)]
    b1 = edge_to_node_matrix(edges, nodes, one_indexed=False).to_sparse()
    b2 = triangle_to_edge_matrix(triangles, edges).to_sparse()

    X0[X0 != 0] = 1

    I = torch.eye(X0.shape[0])

    if X1 is None:
        X1 = torch.tensor(edges)

        # X1_in, X1_out = X0[X1[:, 0]], X0[X1[:, 1]]
        # X1_i_i, X1_o_i = I[X1[:, 0]], I[X1[:, 1]]
        # X1 = torch.logical_and(X1_in, X1_out).float()
        # X1 = torch.logical_or(X1_in, X1_out).float()
        # X1_i = X1_i_i + X1_o_i
        # X1 = X1 - X1_i
        # X1 = (X1_in + X1_out)
        # X1 = torch.sparse.softmax(X1.to_sparse(), dim = 1).to_dense()

    if X2 is None:
        X2 = torch.tensor(triangles)

        # X2_i, X2_j, X2_k = X0[X2[:, 0]], X0[X2[:, 1]], X0[X2[:, 2]]
        # X2_i_i, X2_j_i, X2_k_i = I[X2[:, 0]], I[X2[:, 1]], I[X2[:, 2]]
        # X2 = torch.logical_and(X2_i, torch.logical_and(X2_j, X2_k)).float()
        # X2 = torch.logical_or(X2_i, torch.logical_or(X2_j, X2_k)).float()
        # X2_I = (X2_i_i + X2_j_i + X2_k_i)
        # X2 = X2 - X2_I
        # X2 = torch.sparse.softmax(X2.to_sparse(), dim = 1).to_dense()

        # X2_1 = torch.logical_and(X2_i, X2_j).float()
        # X2_2 = torch.logical_and(X2_i, X2_k).float()
        # X2_3 = torch.logical_and(X2_j, X2_k).float()

        # X2 = torch.logical_or(X2_i, torch.logical_or(X2_j, X2_k)).float()

    return SCData(X0, X1, X2, b1, b2, labels)


def repair_sparse(matrix, ideal_shape):
    # Only use this if last few cols/rows are empty and were removed in sparse operation
    i_x, i_y = ideal_shape
    m_x, m_y = matrix.shape[0], matrix.shape[1]
    indices = matrix.coalesce().indices()
    values = matrix.coalesce().values()
    if i_x > m_x or i_y > m_y:
        additional_i = torch.tensor([[i_x - 1], [i_y - 1]], dtype=torch.float)
        additional_v = torch.tensor([0], dtype=torch.float)
        indices = torch.cat([indices, additional_i], dim=1)
        values = torch.cat([values, additional_v], dim=0)
    return torch.sparse_coo_tensor(indices, values)


def to_sparse_coo(matrix):
    indices = matrix[0:2]
    values = matrix[2:3].squeeze()
    return torch.sparse_coo_tensor(indices, values)

def sparse_diag_identity(n):
    i = [i for i in range(n)]
    return torch.sparse_coo_tensor(torch.tensor([i, i]), torch.ones(n))

def sparse_diag(tensor):
    i = [i for i in range(tensor.shape[0])]
    return torch.sparse_coo_tensor(torch.tensor([i, i]), tensor)


def chebyshev(L, X, k=3):
    if k == 1:
        return torch.sparse.mm(L, X)
    dp = [X, torch.sparse.mm(L, X)]
    for i in range(2, k):
        nxt = 2*(torch.sparse.mm(L, dp[i-1]))
        dp.append(torch.sparse.FloatTensor.add(nxt, -(dp[i-2])))
    return torch.cat(dp, dim=1)


def torch_sparse_to_scipy_sparse(matrix):
    i = matrix.coalesce().indices().cpu()
    v = matrix.coalesce().values().cpu()

    (m, n) = matrix.shape[0], matrix.shape[1]
    return coo_matrix((v, i), shape=(m, n))


def scipy_sparse_to_torch_sparse(matrix):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v)


def normalise(L, half_interval = False):
    M = L.shape[0]
    L = torch_sparse_to_scipy_sparse(L)
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    ret = L.copy()
    if half_interval:
        ret *= 1.0 / topeig
    else:
        ret *= 2.0 / topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)
    ret.setdiag(ret.diagonal(0) - np.ones(M), 0)


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

    I_cat, V_cat = batch_sparse_matrix(L_i_list, L_v_list, sizes, sizes)
    # lapacian_batch = torch.sparse_coo_tensor(L_cat, V_cat)
    batch = [[i for _ in range(sizes[i])] for i in range(len(sizes))]
    batch = torch.tensor([i for sublist in batch for i in sublist])
    return feature_batch, I_cat, V_cat, batch


def batch_sparse_matrix(L_i_list, L_v_list, size_x, size_y):
    L_i_list = list(L_i_list)
    mx_x, mx_y = 0, 0
    for i in range(1, len(L_i_list)):
        mx_x += size_x[i - 1]
        mx_y += size_y[i - 1]
        L_i_list[i][0] += mx_x
        L_i_list[i][1] += mx_y
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



