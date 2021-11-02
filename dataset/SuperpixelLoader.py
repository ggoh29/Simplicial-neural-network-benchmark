import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
from dataset.ImageToDataset import ImageToSimplicialComplex
from constants import DEVICE
from enum import Enum
from skimage import color

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

class SuperPixelLoader(torch.utils.data.Dataset):

	def __init__(self, dataset_name, superpixel_size, simplicial_complex_type, train, batchsize, pool_size, simplicial_complex_size=2):
		'''
		:param dataset_name: Name of dataset. currently either MNIST or CIFAR10.
		:param superpixel_size: Number of superpixel nodes per image
		:param simplicial_complex_type: Format used to construct simplicial complex. Currently either RAG or EdgeFlow.
		:param simplicial_complex_size: Maximum size of simplices in resulting simplicial complex. In range [0, 2]
		'''

		if dataset_name == DatasetType.MNIST:
			dataset = datasets.MNIST
		elif dataset_name == DatasetType.CIFAR10:
			dataset = datasets.CIFAR10

		self.type = dataset_name

		self.data = dataset(root='./data', train=train, download=True, transform=transforms.ToTensor())

		self.data = [*sorted(self.data, key = lambda i : i[1])][:2 * (len(self.data)//5)]
		self.data = make_smaller_dataset_4_classes(self.data)

		self.sc_size = simplicial_complex_size
		self.I2SC = ImageToSimplicialComplex(superpixel_size, simplicial_complex_type, pool_size, simplicial_complex_size)
		self.n_samples = len(self.data)

		self.loader = DataLoader(self.data, batch_size=batchsize, shuffle=True)
		self.loader_iter = iter(self.loader)


	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		image, label = self.train_data[idx]
		# if self.type == datasets.CIFAR10:
		# 	image = color.rgb2gray(image)
		return self.I2SC.image_to_lapacian(image), label

	def __iter__(self):
		return self

	def __next__(self):
		train_features, train_labels = next(self.loader_iter, (None, None))

		if train_features is None:
			self.loader_iter = iter(self.loader)
			raise StopIteration

		train_features.to(DEVICE), train_labels.to(DEVICE)
		X_batch, L_batch, batch_size = self.I2SC.process_batch(train_features)
		return X_batch, L_batch, batch_size, train_labels
