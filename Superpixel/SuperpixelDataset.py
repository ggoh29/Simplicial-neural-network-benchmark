import torch
import torchvision.transforms as transforms
from torch_geometric.data import InMemoryDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from torchvision import datasets
from collections import Counter
from itertools import accumulate

dataset_dct = {datasets.MNIST: "MNIST",
               datasets.CIFAR10: "CIFAR10"}


class SuperpixelSCDataset(InMemoryDataset):

    def __init__(self, root,
                 dataset_name,
                 superpix_size: int,
                 edgeflow_type,
                 processor_type,
                 image_processor,
                 dataset_size: int,
                 n_jobs: int = 8,
                 train: bool = True):
        self.dataset = dataset_name
        self.n_jobs = n_jobs
        self.dataset_size = dataset_size

        self.train_str = {True: "train", False: "test"}[train]
        self.train = train

        self.name = f"{dataset_dct[dataset_name]}_{superpix_size}_{edgeflow_type.__name__}_{processor_type.__class__.__name__}"
        folder = f"{root}/{self.name}/{self.train_str}"

        self.processor_type = processor_type
        self.ImageProcessor = image_processor(superpix_size, edgeflow_type)

        self.pre_transform = lambda image: processor_type.process(self.ImageProcessor.image_to_features(image))

        super().__init__(folder, pre_transform=self.pre_transform)
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
        self.data_download = self.dataset(root='./data', train=self.train, download=True,
                                          transform=transforms.ToTensor())
        data = [*sorted(self.data_download, key=lambda i: i[1])]
        counts = [*map(lambda i: i[1], data)]
        total = len(data)
        if self.dataset_size < total:
            index = []
            counter = Counter(counts)
            for i in range(10):
                index.append(counter[i])

            offset = self.dataset_size  // 10
            assert(offset <= min(index))

            index = [0] + list(accumulate(index))[:-1]

            split = []
            for i in index:
                split += data[i: i + offset]

            self.data_download = split

    @property
    def processed_file_names(self):
        return ["features.pt"]

    def process(self):
        print(f"Pre-transforming {self.train_str} dataset..")
        data_list = Parallel(n_jobs=self.n_jobs, prefer="threads") \
            (delayed(self.pre_transform)(image) for image in tqdm(self.data_download))

        print(f"Finished pre-transforming {self.train_str} dataset.")
        data, slices = self.processor_type.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        return self.processor_type.get(self.data, self.slices, idx)

    def get_name(self):
        return self.name

    def get_val_train_split(self, total=60000, train=55000):
        assert (train < total)
        data = [self.__getitem__(i) for i in range(len(self))]
        assert (len(data) == total)
        data = [*sorted(data, key=lambda i: i.label.item())]

        counter = Counter([*map(lambda i : i.label.item(), data)])
        index = []
        for i in range(10):
            index.append(counter[i])

        offset = (total - train) // 10
        assert (offset <= min(index))

        train_split = []
        val_split = []

        end = list(accumulate(index))
        start = [0] + end[:-1]

        for s, e in zip(start, end):
            val_split += data[s : s + offset]
            train_split += data[s + offset : e]

        return train_split, val_split
