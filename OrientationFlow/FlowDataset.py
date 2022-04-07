import torch
from torch_geometric.data import InMemoryDataset
from OrientationFlow.OrientationFlowGraph import gen_orientation_graph
from joblib import Parallel, delayed
from tqdm import tqdm

class FlowSCDataset(InMemoryDataset):

    def __init__(self, root, processor_type, num_points=600, num_train=1000, num_test=200, train_orient='default', test_orient='random', n_jobs=8):

        self.processor_type = processor_type
        self._num_classes = 2
        self._num_points = num_points
        self._num_train = num_train
        self._num_test = num_test
        self._train_orient = train_orient
        self._test_orient = test_orient
        self._n_jobs = n_jobs
        folder = f"{root}/Flow/{processor_type.__class__.__name__}"

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
        return

    @property
    def processed_file_names(self):
        return ["features.pt"]

    def process(self):
        train, test = gen_orientation_graph(self._num_points, self._num_train, self._num_test,
                                            self._train_orient, self._test_orient, self._n_jobs)

        self.data_download = train + test
        assert (len(train) == self._num_train)
        assert (len(test) == self._num_test)

        print(f"Pre-transforming dataset..")
        data_list = Parallel(n_jobs=self._n_jobs, prefer="threads") \
            (delayed(self.processor_type.process)(image) for image in tqdm(self.data_download))

        print(f"Finished pre-transforming dataset.")
        data, slices = self.processor_type.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        return self.processor_type.get(self.data, self.slices, idx)

    def get_name(self):
        return self.name

    def get_val_train_split(self):
        data = [self.__getitem__(i) for i in range(len(self))]
        train_split = data[:self._num_train]
        val_split = data[self._num_train:]

        assert (len(train_split) == self._num_train)
        assert (len(val_split) == self._num_test)

        return train_split, val_split
