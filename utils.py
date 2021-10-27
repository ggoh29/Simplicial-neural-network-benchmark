import torch
from constants import DEVICE


def edge_to_node_matrix(edges, nodes):
    sigma1 = torch.tensor([[0 for _ in edges] for _ in nodes], dtype=torch.float, device=DEVICE)
    j = 0
    for edge in edges:
        x, y = edge
        sigma1[x - 1][j] = -1
        sigma1[y - 1][j] = 1
        j += 1
    return sigma1

def triangle_to_edge_matrix(triangles, edges):
    sigma2 = torch.tensor([[0 for _ in triangles] for _ in edges], dtype=torch.float, device=DEVICE)

    edges = [e for e in edges]
    edges = {edges[i]: i for i in range(len(edges))}
    for l in range(len(triangles)):
        i, j, k = triangles[l]
        sigma2[edges[(i, j)]][l] = 1
        sigma2[edges[(j, k)]][l] = 1
        sigma2[edges[(i, k)]][l] = -1

    return sigma2