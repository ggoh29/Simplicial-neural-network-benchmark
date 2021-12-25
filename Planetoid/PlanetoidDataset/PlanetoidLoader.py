from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset
import numpy as np
import torch
from models.nn_utils import torch_sparse_to_scipy_sparse, scipy_sparse_to_torch_sparse, to_sparse_coo, convert_to_SC, remove_diag_sparse
import scipy

train_val_test_dct = {'Cora' : (0, 140, 640, 1640), 'CiteSeer' : (0, 120, 620, 1620), 'PubMed' : (0, 60, 560, 1560)}
node_size_dct = {'Cora' : 2708, 'CiteSeer' : 3327, 'PubMed' : 19717}


class PlanetoidSCDataset(InMemoryDataset):

    def __init__(self, root, dataset_name, processor_type):
        self.root = root
        self.dataset_name = dataset_name
        self.processor_type = processor_type

        start, train, val, test = train_val_test_dct[dataset_name]

        nodes = np.array([i for i in range(node_size_dct[dataset_name])])
        np.random.shuffle(nodes)

        self.train_split = nodes[start: train]
        self.val_split = nodes[train: val]
        self.test_split = nodes[val: test]

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
        adj_ones = torch.ones(edges.shape[1])
        adj = torch.sparse_coo_tensor(edges, adj_ones)

        adj = remove_diag_sparse(adj)

        dataset = convert_to_SC(adj, features, labels)
        dataset = [self.processor_type.process(dataset)]

        data, slices = self.processor_type.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

    def get_train(self):
        return self._get_node_subsection(self.train_split)

    def get_val(self):
        return self._get_node_subsection(self.val_split)

    def get_test(self):
        return self._get_node_subsection(self.test_split)

    def get_full(self):
        data_dct = self.get(0)
        data_dct = self.processor_type.batch([data_dct])[0]
        data_dct = self.processor_type.clean_feature_dct(data_dct)
        return self.processor_type.repair(data_dct)

    def _get_node_subsection(self, idx_list):
        dataset = self.__getitem__(0)
        idx_list = torch.tensor(idx_list)
        adj = to_sparse_coo(dataset.L0).to_dense()
        adj = torch.index_select(adj, 0, idx_list)
        adj = torch.index_select(adj, 1, idx_list)
        adj = torch.triu(adj, diagonal=1).to_sparse()
        features = dataset.X0[idx_list]
        labels = dataset.label[idx_list]
        dataset = convert_to_SC(adj, features, labels)
        dataset = self.processor_type.process(dataset)
        data_dct = self.processor_type.batch([dataset])[0]
        data_dct = self.processor_type.clean_feature_dct(data_dct)
        return self.processor_type.repair(data_dct)

    def __getitem__(self, idx):
        return self.processor_type.get(self.data, self.slices, idx)

    def get_name(self):
        return self.name