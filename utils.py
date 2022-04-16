import torch


def sparse_to_tensor(matrix):
    "Converts a sparse matrix to a 3 x N matrix"
    indices = matrix.coalesce().indices()
    values = matrix.coalesce().values().unsqueeze(0)
    return torch.cat([indices, values], dim=0)


def tensor_to_sparse(matrix):
    "Converts a 3 x N matrix to a sparse matrix"
    indices = matrix[0:2]
    values = matrix[2:3].squeeze()
    return torch.sparse_coo_tensor(indices, values)


def ensure_input_is_tensor(input):
    if input.is_sparse:
        input = sparse_to_tensor(input)
    return input


def edge_to_node_matrix(edges, nodes, one_indexed=True):
    sigma1 = torch.ones((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)
    j = 0
    for edge in edges:
        x, y = edge
        sigma1[x - offset][j] -= 1
        sigma1[y - offset][j] += 1
        j += 1
    return sigma1


def triangle_to_edge_matrix(triangles, edges):
    sigma2 = torch.ones((len(edges), len(triangles)), dtype=torch.float)
    edges = [e for e in edges]
    edges = {edges[i]: i for i in range(len(edges))}
    for l in range(len(triangles)):
        i, j, k = triangles[l]
        if (i, j) in edges:
            sigma2[edges[(i, j)]][l] += 1
        else:
            sigma2[edges[(j, i)]][l] -= 1

        if (j, k) in edges:
            sigma2[edges[(j, k)]][l] += 1
        else:
            sigma2[edges[(k, j)]][l] -= 1

        if (i, k) in edges:
            sigma2[edges[(i, k)]][l] -= 1
        else:
            sigma2[edges[(k, i)]][l] += 1

    return sigma2
