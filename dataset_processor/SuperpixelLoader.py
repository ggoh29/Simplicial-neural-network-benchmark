import torch
from dataset_processor.ImageProcessor import ProcessImage, SCData
import torchvision.transforms as transforms
from torch_geometric.data import InMemoryDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from torchvision import datasets

dataset_dct = {datasets.MNIST : "MNIST",
			   datasets.CIFAR10 : "CIFAR10"}


# functions to make a smaller dataset_processor for testing
def make_smaller_dataset_2_classes(data):
	l = len(data)
	data = data[:l // 4] + data[(l // 2):(3 * l // 4)]
	return data


def make_smaller_dataset_4_classes(data):
	l = len(data)
	data = data[:l // 8] + data[(l // 4):(3 * l // 8)] \
		   + data[(l // 2):(5 * l // 8)] + data[(3 * l // 4):(7 * l // 8)]
	return data


class SimplicialComplexDataset(InMemoryDataset):

	def __init__(self, root, dataset_name, superpix_size, edgeflow_type, processor_type, n_jobs=8, train=True):

		self.dataset = dataset_name
		self.n_jobs = n_jobs

		self.train_str = {True : "train", False : "test"}[train]
		self.train = train

		name = f"{dataset_dct[dataset_name]}_{superpix_size}_{edgeflow_type.__name__}_{processor_type.__class__.__name__}/{self.train_str}"
		folder = f"{root}/{name}"

		self.processor_type = processor_type
		self.ImageProcessor = ProcessImage(superpix_size, edgeflow_type)

		def transform(image):
				scData = self.ImageProcessor.image_to_features(image)
				return processor_type.process(scData)

		self.pre_transform = transform

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
		# self.data_download = [*sorted(self.data_download, key=lambda i: i[1])][:(len(self.data_download) // 5)]
		# self.data_download = make_smaller_dataset_2_classes(self.data_download)

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

