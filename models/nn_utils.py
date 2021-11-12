import torch
import scipy.sparse.linalg as spl
import numpy as np
from constants import DEVICE
from scipy.sparse import coo_matrix

def chebyshev(L, X, k=3):
    dp = [X, torch.sparse.mm(L, X)]
    for i in range(2, k):
        nxt = 2*(torch.sparse.mm(L, dp[i-1]))
        dp.append(torch.sparse.FloatTensor.add(nxt, -(dp[i-2])))
    return torch.cat(dp, dim=1)

def normalise(L):

    i = L.coalesce().indices()
    v = L.coalesce().values()
    M = L.shape[0]
    L = coo_matrix((v, i), shape=(M, M))
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    ret = L.copy()
    ret *= 2.0 / topeig
    ret.setdiag(ret.diagonal(0) - np.ones(M), 0)
    values = ret.data
    indices = np.vstack((ret.row, ret.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v)

