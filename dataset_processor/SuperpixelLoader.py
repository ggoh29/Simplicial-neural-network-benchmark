from dataset_processor.DatasetBatcher import DatasetBatcher
from enum import Enum
import torch
from torchvision import datasets
from dataset_processor.ImageProcessor import ProcessImage, SCData
import torchvision.transforms as transforms
from torch_geometric.data import InMemoryDataset
from joblib import Parallel, delayed
from tqdm import tqdm


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


class DatasetType(Enum):
	MNIST = 1
	CIFAR10 = 2


class SimplicialComplexDataset(InMemoryDataset):

	def __init__(self, root, name, superpix_size, edgeflow_type, complex_size=2, n_jobs=4, train=True):

		if name == DatasetType.MNIST:
			self.dataset = datasets.MNIST
		elif name == DatasetType.CIFAR10:
			self.dataset = datasets.CIFAR10

		self.n_jobs = n_jobs

		if train:
			train_str = "Train"
		else:
			train_str = "Test"

		self.train = train

		name = f"{name.name}_{superpix_size}_{edgeflow_type.__name__}/{train_str}"
		folder = f"{root}/{name}"

		self.batcher = DatasetBatcher(complex_size)
		self.ImageProcessor = ProcessImage(superpix_size, edgeflow_type)
		self.pre_transform = self.ImageProcessor.image_to_features

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
		#self.data_download = [*sorted(self.data_download, key=lambda i: i[1])][:2 * (len(self.data_download) // 5)]
		#self.data_download = make_smaller_dataset_4_classes(self.data_download)

	@property
	def processed_file_names(self):
		return ["features.pt"]

	def process(self):
		# Read data into huge `Data` list.
		if self.pre_transform is not None:
			print("Pre-transforming dataset_processor...")
			data_list = Parallel(n_jobs=self.n_jobs, prefer="threads") \
				(delayed(self.pre_transform)(image) for image in tqdm(self.data_download))

		print("Finished Pre-transforming dataset_processor.")
		data, slices = self.collate(data_list)
		torch.save((data, slices), self.processed_paths[0])

	def collate(self, data_list):
		X0, X1, X2 = [], [], []
		sigma1, sigma2 = [], []
		label = []

		x0_total, x1_total, x2_total = 0, 0, 0
		s1_total, s2_total = 0, 0

		slices = {"X0": [0],
				  "X1": [0],
				  "X2": [0],
				  "sigma1": [0],
				  "sigma2": [0]}

		for data in data_list:
			x0, x1, x2 = data.X0, data.X1, data.X2
			s1, s2 = data.sigma1, data.sigma2
			l = data.label
			x0_s, x1_s, x2_s = x0.shape[0], x1.shape[0], x2.shape[0]
			s1_s, s2_s = s1.shape[1], s2.shape[1]

			X0.append(x0);
			X1.append(x1);
			X2.append(x2)
			sigma1.append(s1);
			sigma2.append(s2)
			label.append(l)

			x0_total += x0_s
			x1_total += x1_s
			x2_total += x2_s
			s1_total += s1_s
			s2_total += s2_s

			slices["X0"].append(x0_total)
			slices["X1"].append(x1_total)
			slices["X2"].append(x2_total)
			slices["sigma1"].append(s1_total)
			slices["sigma2"].append(s2_total)

		X0 = torch.cat(X0, dim=0)
		X1 = torch.cat(X1, dim=0)
		X2 = torch.cat(X2, dim=0)
		sigma1 = torch.cat(sigma1, dim=-1)
		sigma2 = torch.cat(sigma2, dim=-1)
		label = torch.cat(label, dim=-1)

		data = SCData(X0, X1, X2, sigma1, sigma2, label)

		return data, slices

	def __getitem__(self, idx):
		return self.get(idx)

	def get(self, idx):
		x0_slice = self.slices["X0"][idx:idx + 2]
		x1_slice = self.slices["X1"][idx:idx + 2]
		x2_slice = self.slices["X2"][idx:idx + 2]
		s1_slice = self.slices["sigma1"][idx:idx + 2]
		s2_slice = self.slices["sigma2"][idx:idx + 2]
		label_slice = [idx, idx + 1]

		X0 = self.data.X0[x0_slice[0]: x0_slice[1]]
		X1 = self.data.X1[x1_slice[0]: x1_slice[1]]
		X2 = self.data.X2[x2_slice[0]: x2_slice[1]]

		sigma1 = self.data.sigma1[:, s1_slice[0]: s1_slice[1]]
		sigma2 = self.data.sigma2[:, s2_slice[0]: s2_slice[1]]

		label = self.data.label[label_slice[0]: label_slice[1]]

		return SCData(X0, X1, X2, sigma1, sigma2, label)

	def batch(self, datalist):
		return self.batcher.collated_data_to_batch(datalist)
