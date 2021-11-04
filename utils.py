import torch
from constants import DEVICE


def dense_to_tensor(matrix):
    "Converts a dense matrix to a 3 x N matrix"
    indices = matrix.coalesce().indices()
    values = matrix.coalesce().values().unsqueeze(0)
    return torch.cat([indices, values], dim=0)


def tensor_to_dense(matrix):
    "Converts a 3 x N matrix to a dense matrix"
    indices = matrix[0:2]
    values = matrix[2:3].squeeze()
    return torch.sparse_coo_tensor(indices, values)


def edge_to_node_matrix(edges, nodes):
    sigma1 = torch.tensor([[0 for _ in edges] for _ in nodes], dtype=torch.float, device=DEVICE)
    j = 0
    for edge in edges:
        x, y = edge
        sigma1[x - 1][j] -= 1
        sigma1[y - 1][j] += 1
        j += 1
    return sigma1

def triangle_to_edge_matrix(triangles, edges):
    sigma2 = torch.tensor([[0 for _ in triangles] for _ in edges], dtype=torch.float, device=DEVICE)

    edges = [e for e in edges]
    edges = {edges[i]: i for i in range(len(edges))}
    for l in range(len(triangles)):
        i, j, k = triangles[l]
        if (i,j) in edges:
            sigma2[edges[(i, j)]][l] += 1
        else:
            sigma2[edges[(j, i)]][l] -= 1

        if (j,k) in edges:
            sigma2[edges[(j, k)]][l] += 1
        else:
            sigma2[edges[(k, j)]][l] -= 1

        if (i,k) in edges:
            sigma2[edges[(i, k)]][l] -= 1
        else:
            sigma2[edges[(k, i)]][l] += 1

    return sigma2