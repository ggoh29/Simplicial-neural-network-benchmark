import torch

def chebyshev(L, X, k=3):
    dp = [X, torch.sparse.mm(L, X)]
    for i in range(2, k):
        nxt = 2*(torch.sparse.mm(L, dp[i-1]))
        dp.append(torch.sparse.FloatTensor.add(nxt, -(dp[i-2])))
    return torch.cat(dp, dim=1)
