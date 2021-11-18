import torch
from models.ProcessorTemplate import NNProcessor
from utils import sparse_to_tensor
from models.nn_utils import to_sparse_coo
from models.nn_utils import batch_all_feature_and_lapacian_pair, convert_indices_and_values_to_sparse, normalise

class SimplicialObject:

	def __init__(self, X0, X1, X2, L0, L1, L2, label):
		self.X0 = X0
		self.X1 = X1
		self.X2 = X2

		# L0, L1 and L2 can either be sparse or dense but since python doesn't really have overloading, doing this instead
		if L0.is_sparse:
			L0 = sparse_to_tensor(L0)
		self.L0 = L0

		if L1.is_sparse:
			L1 = sparse_to_tensor(L1)
		self.L1 = L1

		if L2.is_sparse:
			L2 = sparse_to_tensor(L2)
		self.L2 = L2

		self.label = label


class SNNStefProcessor(NNProcessor):

	def process(self, scData):
		b1, b2 = to_sparse_coo(scData.b1), to_sparse_coo(scData.b2)

		X0, X1, X2 = scData.X0, scData.X1, scData.X2

		L0 = torch.sparse.mm(b1, b1.t())
		L1 = torch.sparse.FloatTensor.add(torch.sparse.mm(b1.t(), b1), torch.sparse.mm(b2, b2.t()))
		L2 = torch.sparse.mm(b2.t(), b2)

		L0 = normalise(L0)
		L1 = normalise(L1)
		L2 = normalise(L2)

		# splitting the sparse tensor as pooling cannot return sparse and to make preparation for minibatching easier
		assert (X0.size()[0] == L0.size()[0])
		assert (X1.size()[0] == L1.size()[0])
		assert (X2.size()[0] == L2.size()[0])

		label = scData.label

		return SimplicialObject(X0, X1, X2, L0, L1, L2, label)

	def collate(self, data_list):
		X0, X1, X2 = [], [], []
		L0, L1, L2 = [], [], []
		label = []

		x0_total, x1_total, x2_total = 0, 0, 0
		l0_total, l1_total, l2_total = 0, 0, 0

		slices = {"X0": [0],
							"X1": [0],
							"X2": [0],
							"L0": [0],
							"L1": [0],
							"L2": [0]}

		for data in data_list:
			x0, x1, x2 = data.X0, data.X1, data.X2
			l0, l1, l2 = data.L0, data.L1, data.L2
			l = data.label
			x0_s, x1_s, x2_s = x0.shape[0], x1.shape[0], x2.shape[0]
			l0_s, l1_s, l2_s = l0.shape[1], l1.shape[1], l2.shape[1]

			X0.append(x0)
			X1.append(x1)
			X2.append(x2)
			L0.append(l0)
			L1.append(l1)
			L2.append(l2)
			label.append(l)

			x0_total += x0_s
			x1_total += x1_s
			x2_total += x2_s
			l0_total += l0_s
			l1_total += l1_s
			l2_total += l2_s

			slices["X0"].append(x0_total)
			slices["X1"].append(x1_total)
			slices["X2"].append(x2_total)
			slices["L0"].append(l0_total)
			slices["L1"].append(l1_total)
			slices["L2"].append(l2_total)

		X0 = torch.cat(X0, dim=0).to('cpu')
		X1 = torch.cat(X1, dim=0).to('cpu')
		X2 = torch.cat(X2, dim=0).to('cpu')
		L0 = torch.cat(L0, dim=-1).to('cpu')
		L1 = torch.cat(L1, dim=-1).to('cpu')
		L2 = torch.cat(L2, dim=-1).to('cpu')
		label = torch.cat(label, dim=-1).to('cpu')

		data = SimplicialObject(X0, X1, X2, L0, L1, L2, label)

		return data, slices

	def get(self, data, slices , idx):
		x0_slice = slices["X0"][idx:idx + 2]
		x1_slice = slices["X1"][idx:idx + 2]
		x2_slice = slices["X2"][idx:idx + 2]
		l0_slice = slices["L0"][idx:idx + 2]
		l1_slice = slices["L1"][idx:idx + 2]
		l2_slice = slices["L2"][idx:idx + 2]
		label_slice = [idx, idx + 1]

		X0 = data.X0[x0_slice[0]: x0_slice[1]]
		X1 = data.X1[x1_slice[0]: x1_slice[1]]
		X2 = data.X2[x2_slice[0]: x2_slice[1]]

		L0 = data.L0[:, l0_slice[0]: l0_slice[1]]
		L1 = data.L1[:, l1_slice[0]: l1_slice[1]]
		L2 = data.L2[:, l2_slice[0]: l2_slice[1]]

		label = data.label[label_slice[0]: label_slice[1]]

		return SimplicialObject(X0, X1, X2, L0, L1, L2, label)

	def batch(self, objectList):
		def unpack_SimplicialObject(SimplicialObject):
			X0, X1, X2 = SimplicialObject.X0, SimplicialObject.X1, SimplicialObject.X2
			L0, L1, L2 = SimplicialObject.L0, SimplicialObject.L1, SimplicialObject.L2

			L0_i, L0_v = L0[0:2], L0[2:3].squeeze()
			L1_i, L1_v = L1[0:2], L1[2:3].squeeze()
			L2_i, L2_v = L2[0:2], L2[2:3].squeeze()

			label = SimplicialObject.label
			return [X0, X1, X2], [L0_i, L1_i, L2_i], [L0_v, L1_v, L2_v], label

		unpacked_grapObject = [unpack_SimplicialObject(g) for g in objectList]
		X, L_i, L_v, labels = [*zip(*unpacked_grapObject)]
		X, L_i, L_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)]

		features_dct = batch_all_feature_and_lapacian_pair(X, L_i, L_v)

		labels = torch.cat(labels, dim=0)
		return features_dct, labels

	def clean_feature_dct(self, feature_dct):
		return convert_indices_and_values_to_sparse(feature_dct, 'lapacian_indices', 'lapacian_values', 'lapacian')