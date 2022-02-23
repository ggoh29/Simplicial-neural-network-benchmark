from torch_geometric.datasets import Planetoid
import torch
import networkx as nx
from models.nn_utils import convert_to_SC, to_sparse_coo
from utils import edge_to_node_matrix, triangle_to_edge_matrix


class GraphObject:

    def __init__(self, x, edge_index, y, train_mask, val_mask, test_mask):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

def gen_dataset():
    data = Planetoid('./data', "Cora")[0]

    edges = data.edge_index
    n = data.x.shape[0]
    adj = torch.zeros((n, n))
    adj = adj.index_put_(tuple(edges), torch.ones(1))
    adj = torch.triu(adj)
    edges = torch.nonzero(adj).tolist()
    nodes = [i for i in range(n)]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    triangles = [list(sorted(x)) for x in nx.enumerate_all_cliques(g) if len(x) == 3]
    quads = [list(sorted(x)) for x in nx.enumerate_all_cliques(g) if len(x) == 4]
    quads_indices_set = set()

    labels = [0 for _ in range(n)]
    for quad in quads:
        for index in quad:
            labels[index] = 2
            quads_indices_set.add(index)

    tri_indices_set = set()

    for triangle in triangles:
        for index in triangle:
            if index not in quads_indices_set:
                labels[index] = 1
                tri_indices_set.add(index)

    y = torch.tensor(labels)

    train_mask = []
    val_mask = []
    test_mask = []

    train_no = 60
    val_no = 300
    test_no = 1000

    class_1 = 0
    class_2 = 0
    class_3 = 0

    val = 0
    test = 0

    for i in nodes:
        if class_1 + class_2 + class_3 < train_no:
            if i in tri_indices_set and class_1 < 20:
                train_mask.append(i)
                class_1 += 1
            elif i in quads_indices_set and class_2 < 20:
                train_mask.append(i)
                class_2 += 1
            elif class_3 < 20:
                train_mask.append(i)
                class_3 += 1
        elif val < val_no:
            val_mask.append(i)
            val += 1
        elif test < test_no:
            test_mask.append(i)
            test += 1

    train_index = torch.tensor(train_mask)
    train_mask = torch.zeros(n)
    train_mask.index_fill_(0, train_index, 1)
    train_mask = train_mask > 0

    test_index = torch.tensor(test_mask)
    test_mask = torch.zeros(n)
    test_mask.index_fill_(0, test_index, 1)
    test_mask = test_mask > 0

    val_index = torch.tensor(val_mask)
    val_mask = torch.zeros(n)
    val_mask.index_fill_(0, val_index, 1)
    val_mask = val_mask > 0

    X0 = torch.sum(adj, dim = 1)
    X0 = torch.nn.functional.one_hot(X0.long()).float()
    # X0 = adj + adj.T
    X0 = torch.ones((adj.shape[0], adj.shape[0]))
    edge_index = torch.nonzero(adj).T

    assert (X0.shape[0] == y.shape[0])
    assert (y.shape[0] == train_mask.shape[0])
    assert (y.shape[0] == val_mask.shape[0])
    assert (y.shape[0] == test_mask.shape[0])

    g = GraphObject(X0, edge_index, y, train_mask, val_mask, test_mask)
    return g



if __name__ == "__main__":
    gen_dataset()
