from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset
import networkx as nx
import numpy as np
from utils import edge_to_node_matrix, triangle_to_edge_matrix
import functools
from models.SCData import SCData
import torch


def get_features(features, sc_list):
    def _get_features(features, sc):
        f = [features[i] for i in sc]
        return functools.reduce(lambda a, b: a + b, f)

    return torch.stack([_get_features(features, sc) for sc in sc_list], dim = 0)


def convert_to_SC(ones, edges, features, labels, reduce = True):
    n = features.shape[0]
    full_adj = torch.empty((n, n), dtype=torch.float)
    adj = torch.sparse_coo_tensor(edges, ones).to_dense()
    full_adj[: adj.shape[0], : adj.shape[1]] = adj
    valid_features = torch.tensor(np.unique(edges), dtype=torch.int)
    if reduce:
        X0 = torch.index_select(features, 0, valid_features)
        adj = torch.index_select(full_adj, 0, valid_features)
        adj = torch.index_select(adj, 1, valid_features)
    else:
        X0 = features
        adj = full_adj
    adj = torch.triu(adj, diagonal=1)
    nodes = [i for i in range(X0.shape[0])]
    edges = adj.to_sparse().coalesce().indices().tolist()

    g = nx.Graph()
    g.add_nodes_from(nodes)
    edges = [(i, j) for i, j in zip(edges[0], edges[1])]
    g.add_edges_from(edges)
    triangles = [x for x in nx.enumerate_all_cliques(g) if len(x) == 3]

    b1 = edge_to_node_matrix(edges, nodes).to_sparse()
    b2 = triangle_to_edge_matrix(triangles, edges).to_sparse()
    X1 = get_features(features, edges)
    X2 = get_features(features, triangles)

    labels = torch.index_select(labels, 0, valid_features)
    return SCData(X0, X1, X2, b1, b2, labels)


class PlanetoidSCDataset(InMemoryDataset):

    def __init__(self, root, dataset_name, processor_type, n_jobs=8):
        self.root = root
        self.dataset_name = dataset_name
        self.n_jobs = n_jobs
        self.processor_type = processor_type

        self.train_split = 0.10
        self.val_split = 0.15
        self.test_split = 0.20

        folder = f"{root}/{self.dataset_name}/{processor_type.__class__.__name__}"

        super().__init__(folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices["X0"]) - 1

    def load_dataset(self):
        """Load the dataset_processor from here and process it if it doesn't exist"""
        print("Loading dataset_processor from disk...")
        data, slices = torch.load(self.processed_paths[0])
        return data, slices

    @property
    def raw_file_names(self):
        return []

    def download(self):
        # Instantiating this will download and process the graph dataset_processor.
        self.data_download = Planetoid(self.root, self.dataset_name)[0]

    @property
    def processed_file_names(self):
        return ["features.pt"]

    def process(self):
        data = self.data_download
        features, edges, labels = data.x, data.edge_index, data.y
        n = edges.shape[1]

        edge_index = np.array(edges.T)
        np.random.shuffle(edge_index)

        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        test_end = int(n * (self.train_split + self.val_split + self.test_split))

        train_edges = edge_index[:train_end].T
        val_edges = edge_index[train_end:val_end].T
        test_edges = edge_index[val_end:test_end].T

        train_ones = torch.ones(train_end)
        val_ones = torch.ones(val_end - train_end)
        test_ones = torch.ones(test_end - val_end)

        train = convert_to_SC(train_ones, train_edges, features, labels)
        val = convert_to_SC(val_ones, val_edges, features, labels)
        test = convert_to_SC(test_ones, test_edges, features, labels)

        data_list = [train, val, test]
        data_list = [self.processor_type.process(data) for data in data_list]

        data, slices = self.processor_type.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_train(self):
        return self.__getitem__(0)

    def get_val(self):
        return self.__getitem__(1)

    def get_test(self):
        return self.__getitem__(2)

    def __getitem__(self, idx):
        return self.processor_type.get(self.data, self.slices, idx)

    def get_name(self):
        return self.name