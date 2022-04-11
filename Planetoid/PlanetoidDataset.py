from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset
import numpy as np
import torch
from models.nn_utils import convert_to_CoChain, remove_diag_sparse, to_sparse_coo, normalise_boundary
from Planetoid.FakeDataset import gen_dataset


class PlanetoidSCDataset(InMemoryDataset):

    def __init__(self, root, dataset_name, processor_type):
        self.root = root
        self.dataset_name = dataset_name
        self.processor_type = processor_type

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
        if self.dataset_name == 'fake':
            self.data_download = gen_dataset()
        else:
            self.data_download = Planetoid(self.root, self.dataset_name)[0]
        nodes = self.data_download.x.shape[0]
        self.nodes = np.array([i for i in range(nodes)])

        self.test_split = self.data_download.test_mask
        self.train_split = self.data_download.train_mask
        self.val_split = self.data_download.val_mask

    @property
    def processed_file_names(self):
        return ["features.pt"]

    def process(self):
        data = self.data_download
        features, edges, labels = data.x, data.edge_index, data.y
        adj_ones = torch.ones(edges.shape[1])
        adj = torch.sparse_coo_tensor(edges, adj_ones)

        # features = preprocess_features(features)
        adj = remove_diag_sparse(adj)
        dataset = convert_to_CoChain(adj, features, labels)
        dataset = [self.processor_type.process(dataset)]
        data, slices = self.processor_type.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

    def get_boundary(self, edge_list, features):
        adj_ones = torch.ones(edge_list.shape[1])
        adj = torch.sparse_coo_tensor(edge_list, adj_ones)

        # features = preprocess_features(features)
        adj = remove_diag_sparse(adj)
        cochain = convert_to_CoChain(adj, features, None)
        b1, b2 = normalise_boundary(cochain.b1, cochain.b2)
        return b1, b2

    def _get_node_subsection(self, idx_list):
        dataset = self.__getitem__(0)
        idx_list = torch.tensor(idx_list)
        adj = to_sparse_coo(dataset.L0).to_dense()
        adj = torch.index_select(adj, 0, idx_list)
        adj = torch.index_select(adj, 1, idx_list)
        adj = torch.triu(adj, diagonal=1).to_sparse()
        features = dataset.X0[idx_list]
        labels = dataset.label[idx_list]
        simplicialComplex = convert_to_CoChain(adj, features, labels)
        simplicialComplex = self.processor_type.process(simplicialComplex)
        simplicialComplex = self.processor_type.batch([simplicialComplex])[0]
        simplicialComplex = self.processor_type.clean_features(simplicialComplex)
        return simplicialComplex

    def get_full(self):
        simplicialComplex = self.get(0)
        simplicialComplex = self.processor_type.batch([simplicialComplex])[0]
        simplicialComplex = self.processor_type.clean_features(simplicialComplex)
        simplicialComplex = self.processor_type.repair(simplicialComplex)
        b1, b2 = self.get_boundary(simplicialComplex.L0.coalesce().indices(), simplicialComplex.X0)
        return simplicialComplex, b1, b2

    def get_train_labels(self):
        simplicialComplex = self.get(0)
        return simplicialComplex.label[self.train_split]

    def get_val_labels(self):
        simplicialComplex = self.get(0)
        return simplicialComplex.label[self.val_split]

    def get_test_labels(self):
        simplicialComplex = self.get(0)
        return simplicialComplex.label[self.test_split]

    def get_labels(self):
        simplicialComplex = self.get(0)
        return simplicialComplex.label

    def get_train_embeds(self, embeds):
        return embeds[self.train_split]

    def get_val_embeds(self, embeds):
        return embeds[self.val_split]

    def get_test_embeds(self, embeds):
        return embeds[self.test_split]

    def __getitem__(self, idx):
        return self.processor_type.get(self.data, self.slices, idx)

    def get_name(self):
        return self.name