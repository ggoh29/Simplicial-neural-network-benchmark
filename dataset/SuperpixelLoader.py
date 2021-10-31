import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
from dataset.ImageToDataset import ImageToSimplicialComplex
import numpy as np
from constants import DEVICE


def prepare_for_mini_batching(x_list, L_i_list, l_v_list):
	feature_batch = torch.cat(x_list, dim=0)

	sizes = [*map(lambda x: x.size()[0], x_list)]
	L_i_list = list(L_i_list)
	mx = 0
	for i in range(1, len(sizes)):
		mx += sizes[i - 1]
		L_i_list[i] += mx
	L_cat = torch.cat(L_i_list, dim=1)
	V_cat = torch.cat(l_v_list, dim=0)
	lapacian_batch = torch.sparse_coo_tensor(L_cat, V_cat)
	batch = [[i for _ in range(sizes[i])] for i in range(len(sizes))]
	batch = torch.tensor([i for sublist in batch for i in sublist], device=DEVICE)
	return feature_batch, lapacian_batch, batch


def batch(samples, sc_size):
	print(samples)
	graphs, labels = map(list, zip(*samples))
	labels = torch.tensor(np.array(labels))
	X, L_i, L_v = [*zip(*graphs)]
	X, L_i, L_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)]

	X_batch, L_batch, batch_size = [], [], []
	for i in range(sc_size + 1):
		x, l, batch = prepare_for_mini_batching(X[i], L_i[i], L_v[i])
		X_batch.append(x)
		L_batch.append(l)
		batch_size.append(batch)

	del X, L_v, L_i

	return (X_batch, L_batch, batch_size), labels

class SuperPixel(torch.utils.data.Dataset):

	def __init__(self, dataset_name, superpixel_size, simplicial_complex_type, train, simplicial_complex_size = 2):
		'''

		:param dataset_name: Name of dataset. currently either MNIST or CIFAR10.
		:param superpixel_size: Number of superpixel nodes per image
		:param simplicial_complex_type: Format used to construct simplicial complex. Currently either RAG or EdgeFlow.
		:param simplicial_complex_size: Maximum size of simplices in resulting simplicial complex. In range [0, 2]
		'''

		if dataset_name == 'MNIST':
			dataset = datasets.MNIST
		elif dataset_name == 'CIFAR10':
			dataset = datasets.CIFAR10

		self.train_data = dataset(root='./data', train=train, download=True, transform=transforms.ToTensor())

		self.sc_size = simplicial_complex_size
		self.I2SC = ImageToSimplicialComplex(superpixel_size, simplicial_complex_type, simplicial_complex_size)

		self.n_samples = len(self.train_data)

	def prepare_for_mini_batching(x_list, L_i_list, l_v_list):
		feature_batch = torch.cat(x_list, dim=0)

		sizes = [*map(lambda x: x.size()[0], x_list)]
		L_i_list = list(L_i_list)
		mx = 0
		for i in range(1, len(sizes)):
			mx += sizes[i - 1]
			L_i_list[i] += mx
		L_cat = torch.cat(L_i_list, dim=1)
		V_cat = torch.cat(l_v_list, dim=0)
		lapacian_batch = torch.sparse_coo_tensor(L_cat, V_cat)
		batch = [[i for _ in range(sizes[i])] for i in range(len(sizes))]
		batch = torch.tensor([i for sublist in batch for i in sublist], device=DEVICE)
		return feature_batch, lapacian_batch, batch

	def batch(self, samples):
		print(samples)
		graphs, labels = map(list, zip(*samples))
		labels = torch.tensor(np.array(labels))
		X, L_i, L_v = [*zip(*graphs)]
		X, L_i, L_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)]

		X_batch, L_batch, batch_size = [], [], []
		for i in range(self.sc_size + 1):
			x, l, batch = self.prepare_for_mini_batching(X[i], L_i[i], L_v[i])
			X_batch.append(x)
			L_batch.append(l)
			batch_size.append(batch)

		del X, L_v, L_i

		return (X_batch, L_batch, batch_size), labels


	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		image, label = self.train_data[idx]
		return self.I2SC.image_to_lapacian(image), label

	def get_train_loader(self, batch_size, shuffle=True):
		return DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x : batch(x, self.sc_size), num_workers=4)

	def get_test_loader(self, batch_size, shuffle=True):
		return DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x : batch(x, self.sc_size), num_workers= 4)


	# def _prepare(self):
	# 	print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
	# 	self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
	# 	for index, sample in enumerate(self.sp_data):
	# 		mean_px, coord = sample[:2]
	#
	# 		try:
	# 			coord = coord / self.img_size
	# 		except AttributeError:
	# 			VOC_has_variable_image_sizes = True
	#
	# 		if self.use_mean_px:
	# 			A = compute_adjacency_matrix_images(coord, mean_px)  # using super-pixel locations + features
	# 		else:
	# 			A = compute_adjacency_matrix_images(coord, mean_px, False)  # using only super-pixel locations
	# 		edges_list, edge_values_list = compute_edges_list(A)  # NEW
	#
	# 		N_nodes = A.shape[0]
	#
	# 		mean_px = mean_px.reshape(N_nodes, -1)
	# 		coord = coord.reshape(N_nodes, 2)
	# 		x = np.concatenate((mean_px, coord), axis=1)
	#
	# 		edge_values_list = edge_values_list.reshape(-1)  # NEW # TO DOUBLE-CHECK !
	#
	# 		self.node_features.append(x)
	# 		self.edge_features.append(edge_values_list)  # NEW
	# 		self.Adj_matrices.append(A)
	# 		self.edges_lists.append(edges_list)
	#
	# 	for index in range(len(self.sp_data)):
	# 		g = dgl.DGLGraph()
	# 		g.add_nodes(self.node_features[index].shape[0])
	# 		g.ndata['feat'] = torch.Tensor(self.node_features[index]).half()
	#
	# 		for src, dsts in enumerate(self.edges_lists[index]):
	# 			# handling for 1 node where the self loop would be the only edge
	# 			# since, VOC Superpixels has few samples (5 samples) with only 1 node
	# 			if self.node_features[index].shape[0] == 1:
	# 				g.add_edges(src, dsts)
	# 			else:
	# 				g.add_edges(src, dsts[dsts != src])
	#
	# 		# adding edge features for Residual Gated ConvNet
	# 		edge_feat_dim = g.ndata['feat'].shape[1]  # dim same as node feature dim
	# 		# g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim).half()
	# 		g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1).half()  # NEW
	#
	# 		self.graph_lists.append(g)
	#
	#
	# def __len__(self):
	# 	"""Return the number of graphs in the dataset."""
	# 	return self.n_samples
	#
	#
	# def __getitem__(self, idx):
	# 	"""
	# 		Get the idx^th sample.
	# 		Parameters
	# 		---------
	# 		idx : int
	# 			The sample index.
	# 		Returns
	# 		-------
	# 		(dgl.DGLGraph, int)
	# 			DGLGraph with node feature stored in `feat` field
	# 			And its label.
	# 	"""
	# 	return self.graph_lists[idx], self.graph_labels[idx]