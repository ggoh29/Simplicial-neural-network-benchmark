import torch
from Superpixel.SuperpixelDataset.ImageProcessor import ProcessImage
import torchvision.transforms as transforms
from torch_geometric.data import InMemoryDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from torchvision import datasets

dataset_dct = {datasets.MNIST: "MNIST",
               datasets.CIFAR10: "CIFAR10"}

def make_smaller_dataset_10_classes(data):
    l = len(data)
    data = [*sorted(data, key=lambda i: i[1])]
    data_out = []
    for i in range(10):
        data_out += data[i * l // 10: (i * 5 + 1) * l // 50]
    return data_out


class SuperpixelSCDataset(InMemoryDataset):

    def __init__(self, root, dataset_name, superpix_size, edgeflow_type, processor_type, n_jobs=8, train=True):
        self.dataset = dataset_name
        self.n_jobs = n_jobs

        self.train_str = {True: "train", False: "test"}[train]
        self.train = train

        self.name = f"{dataset_dct[dataset_name]}_{superpix_size}_{edgeflow_type.__name__}_{processor_type.__class__.__name__}"
        folder = f"{root}/{self.name}/{self.train_str}"

        self.processor_type = processor_type
        self.ImageProcessor = ProcessImage(superpix_size, edgeflow_type)

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
        self.data_download = self.dataset(root='../data', train=self.train, download=True,
                                          transform=transforms.ToTensor())
        self.data_download = make_smaller_dataset_10_classes(self.data_download)

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