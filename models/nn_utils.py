import torch
import scipy
import scipy.sparse.linalg as spl
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from utils import edge_to_node_matrix, triangle_to_edge_matrix
from models.CoChain import CoChain
import functools
import scipy.sparse as sp
from scipy import sparse


def normalise_boundary(b1, b2):
    B1, B2 = to_sparse_coo(b1), to_sparse_coo(b2)
    x0, x1 = B1.shape
    _, x2 = B2.shape

    B1_v_abs, B1_i = torch.abs(B1.coalesce().values()), B1.coalesce().indices()
    B1_sum = torch.sparse.sum(torch.sparse_coo_tensor(B1_i, B1_v_abs, (x0, x1)), dim=1)
    B1_sum_values = B1_sum.to_dense()
    B1_sum_indices = torch.tensor([i for i in range(x0)])
    d0_diag_indices = torch.stack([B1_sum_indices, B1_sum_indices], dim=0)
    B1_sum_inv_values = torch.nan_to_num(1. / B1_sum_values, nan=0., posinf=0., neginf=0.)

    D1_inv = torch.sparse_coo_tensor(d0_diag_indices, 0.5 * B1_sum_inv_values, (x0, x0))
    D3_values = (1 / 3.) * torch.ones(B2.shape[1])
    D3_indices = [i for i in range(B2.shape[1])]
    D3_indices = torch.tensor([D3_indices, D3_indices])
    D3 = torch.sparse_coo_tensor(D3_indices, D3_values, (x2, x2))

    B2D3 = torch.sparse.mm(B2, D3)
    D1invB1 = (1 / np.sqrt(2.)) * torch.sparse.mm(D1_inv, B1)

    return D1invB1, B2D3


def _normalise_boundary(b1, b2):
    B1, B2 = to_sparse_coo(b1), to_sparse_coo(b2)
    _, x2 = B2.shape
    B1, B2 = torch_sparse_to_scipy_sparse(B1), torch_sparse_to_scipy_sparse(B2)

    B1_sum = np.abs(B1).sum(axis=1)
    B1_sum_inv = 1. / B1_sum
    B1_sum_inv[np.isinf(B1_sum_inv) | np.isneginf(B1_sum_inv)] = 0
    D1_inv = sparse.diags((B1_sum_inv * 0.5).A.reshape(-1), 0)
    D3 = (1 / 3.) * sparse.identity(n=B2.shape[1])

    B2D3 = B2 @ D3
    D1invB1 = (1 / np.sqrt(2.)) * D1_inv @ B1

    return scipy_sparse_to_torch_sparse(sparse.coo_matrix(D1invB1)), scipy_sparse_to_torch_sparse(sparse.coo_matrix(B2D3))


def preprocess_features(features):
    """Row-normalize feature matrix"""
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
        return torch.stack(features, dim=0)
    else:
        return torch.tensor([])


def filter_simplices(node_features, simplice):
    s = [node_features[i] for i in simplice]
    common_features = functools.reduce(lambda a, b: torch.logical_and(a, b), s).float()
    return torch.sum(common_features).item() > 0


def correct_orientation(L, up_or_down):
    """
    L : n * n sparse Laplacian matrix
    up_or_down : int in {-1, 1}
    """
    # Add 2 to identity
    identity = 2 * torch.ones(L.shape[0])
    identity_indices = torch.arange(L.shape[0])
    identity_indices = torch.stack([identity_indices, identity_indices], dim=0)
    sparse_identity = torch.sparse_coo_tensor(identity_indices, identity)
    adj = L + sparse_identity

    indices = adj.coalesce().indices()
    values = adj.coalesce().values() * up_or_down
    values[values < -1] = 1
    values = torch.sign(values)

    return torch.sparse_coo_tensor(indices, values)


def convert_to_CoChain(adj, features, labels, X1=None, X2=None):
    X0 = features

    nodes = [i for i in range(X0.shape[0])]
    edges = adj.coalesce().indices().tolist()
    edges = [(i, j) for i, j in zip(edges[0], edges[1])]
    # edges = [*filter(lambda x: filter_simplices(features, x), edges)]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    triangles = [list(sorted(x)) for x in nx.enumerate_all_cliques(g) if len(x) == 3]
    # triangles = [*filter(lambda x: filter_simplices(features, x), triangles) ]
    b1 = edge_to_node_matrix(edges, nodes, one_indexed=False).to_sparse()
    b2 = triangle_to_edge_matrix(triangles, edges).to_sparse()

    if X1 is None:
        X1 = torch.tensor(edges)

    if X2 is None:
        X2 = torch.tensor(triangles)

    return CoChain(X0, X1, X2, b1, b2, labels)


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
        nxt = 2 * (torch.sparse.mm(L, dp[i - 1]))
        dp.append(torch.sparse.FloatTensor.add(nxt, -(dp[i - 2])))
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


def normalise(L):
    M = L.shape[0]
    L = torch_sparse_to_scipy_sparse(L)
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    ret = L.copy()
    ret *= 2.0 / topeig
    ret.setdiag(np.ones(M) - ret.diagonal(0), 0)
    return scipy_sparse_to_torch_sparse(ret)


def batch_all_feature_and_lapacian_pair(X, L_i, L_v):
    X_batch, I_batch, V_batch, batch_index = [], [], [], []
    for i in range(len(X)):
        x, i, v, batch = batch_feature_and_lapacian_pair(X[i], L_i[i], L_v[i])
        X_batch.append(x)
        I_batch.append(i)
        V_batch.append(v)
        batch_index.append(batch)

    features_dct = {'features': X_batch,
                    'lapacian_indices': I_batch,
                    'lapacian_values': V_batch,
                    'batch_index': batch_index}

    # I_batch and V_batch form the indices and values of coo_sparse tensor but sparse tensors
    # cant be stored so storing them as two separate tensors
    return features_dct


def batch_feature_and_lapacian_pair(x_list, L_i_list, L_v_list):
    feature_batch = torch.cat(x_list, dim=0)
    sizes = [*map(lambda x: x.size()[0], x_list)]

    I_cat, V_cat = batch_sparse_matrix(L_i_list, L_v_list, sizes, sizes)
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
