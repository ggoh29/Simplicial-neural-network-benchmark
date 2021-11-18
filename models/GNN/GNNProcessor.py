from utils import sparse_to_tensor
import torch
from models.nn_utils import to_sparse_coo
from models.ProcessorTemplate import NNProcessor
from models.nn_utils import batch_feature_and_lapacian_pair, convert_indices_and_values_to_sparse, normalise

class GraphObject:

	def __init__(self, X0, L0, label):
		self.X0 = X0

		if L0.is_sparse:
			L0 = sparse_to_tensor(L0)
		self.L0 = L0

		self.label = label


class GNNProcessor(NNProcessor):

	def process(self, scData):
		b1, b2 = to_sparse_coo(scData.b1), to_sparse_coo(scData.b2)

		X0 = scData.X0
		L0 = normalise(torch.sparse.mm(b1, b1.t()))
		label = scData.label

		return GraphObject(X0, L0, label)

	def collate(self, data_list):
		X0 = []
		L0 = []
		label = []

		X0_total = 0
		L0_total = 0

		slices = {"X0": [0],
							"L0": [0]}

		for data in data_list:
			x0 = data.X0
			l0 = data.L0
			l = data.label
			x0_s = x0.shape[0]
			l0_s = l0.shape[1]

			X0.append(x0)
			L0.append(l0)
			label.append(l)

			X0_total += x0_s
			L0_total += l0_s

			slices["X0"].append(X0_total)
			slices["L0"].append(L0_total)

		X0 = torch.cat(X0, dim=0).to('cpu')
		L0 = torch.cat(L0, dim=-1).to('cpu')
		label = torch.cat(label, dim=-1).to('cpu')

		data = GraphObject(X0, L0, label)

		return data, slices

	def get(self, data, slices, idx):
		x0_slice = slices["X0"][idx:idx + 2]
		l0_slice = slices["L0"][idx:idx + 2]
		label_slice = [idx, idx + 1]

		X0 = data.X0[x0_slice[0]: x0_slice[1]]
		L0 = data.L0[:, l0_slice[0]: l0_slice[1]]
		label = data.label[label_slice[0]: label_slice[1]]
		return GraphObject(X0, L0, label)

	def batch(self, objectList):
		def unpack_graphObject(graphObject):
			features = graphObject.X0
			lapacian = graphObject.L0
			indices = lapacian[0:2]
			values = lapacian[2:3].squeeze()
			label = graphObject.label
			return features, indices, values, label

		unpacked_grapObject = [unpack_graphObject(g) for g in objectList]
		X, L_i, L_v, labels = [*zip(*unpacked_grapObject)]
		X, L_i, L_v = list(X), list(L_i), list(L_v)
		X_batch, I_batch, V_batch, batch_index = batch_feature_and_lapacian_pair(X, L_i, L_v)
		features_dct = {'features': [X_batch],
										'lapacian_indices': [I_batch],
										'lapacian_values': [V_batch],
										'batch_index': [batch_index]}

		labels = torch.cat(labels, dim=0)
		return features_dct, labels

	def clean_feature_dct(self, feature_dct):
		return convert_indices_and_values_to_sparse(feature_dct, 'lapacian_indices', 'lapacian_values', 'lapacian')