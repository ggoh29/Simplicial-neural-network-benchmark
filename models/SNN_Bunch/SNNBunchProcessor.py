from utils import sparse_to_tensor
import torch
import numpy as np
from models.ProcessorTemplate import NNProcessor
from models.nn_utils import convert_indices_and_values_to_sparse, normalise,\
	batch_sparse_matrix, batch_all_feature_and_lapacian_pair

class SimplicialObject:

	def __init__(self, X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv, label):
		self.X0 = X0
		self.X1 = X1
		self.X2 = X2

		if L0.is_sparse:
			L0 = sparse_to_tensor(L0)
		self.L0 = L0

		if L1.is_sparse:
			L1 = sparse_to_tensor(L1)
		self.L1 = L1

		if L2.is_sparse:
			L2 = sparse_to_tensor(L2)
		self.L2 = L2

		if B2D3.is_sparse:
			B2D3 = sparse_to_tensor(B2D3)
		self.B2D3 = B2D3

		if D2B1TD1inv.is_sparse:
			D2B1TD1inv = sparse_to_tensor(D2B1TD1inv)
		self.D2B1TD1inv = D2B1TD1inv

		if D1invB1.is_sparse:
			D1invB1 = sparse_to_tensor(D1invB1)
		self.D1invB1 = D1invB1

		if B2TD2inv.is_sparse:
			B2TD2inv = sparse_to_tensor(B2TD2inv)
		self.B2TD2inv = B2TD2inv

		self.label = label


class SNNBunchProcessor(NNProcessor):

	def process(self, scData):
		def to_dense(matrix):
			indices = matrix[0:2]
			values = matrix[2:3].squeeze()
			return torch.sparse_coo_tensor(indices, values).to_dense()

		B1, B2 = to_dense(scData.b1).to('cpu'), to_dense(scData.b2).to('cpu')
		X0, X1, X2 = scData.X0.to('cpu'), scData.X1.to('cpu'), scData.X2.to('cpu')
		label = scData.label

		L0 = B1 @ B1.T
		B1_sum = torch.sum(torch.abs(B1), 1)
		d0 = torch.diag(B1_sum)
		B1_sum_inv = torch.nan_to_num(1. / B1_sum, nan=0., posinf=0., neginf=0.)
		d0_inv = torch.diag(B1_sum_inv)
		L0 = d0_inv @ L0
		L0_factor = -1 * torch.diag(1 / (B1_sum_inv + 1))
		L0bias = torch.eye(n=d0.shape[0])
		L0 = L0_factor @ L0 + L0bias

		D1_inv = torch.diag(0.5 * B1_sum_inv)
		D2diag = torch.sum(torch.abs(B2), 1)
		D2diag = torch.maximum(D2diag, torch.tensor([1 for _ in range(D2diag.shape[0])]))
		D2 = torch.diag(D2diag)
		D2_inv = torch.diag(1 / D2diag)
		D3 = (1 / 3.) * torch.eye(n=B2.shape[1])

		# might need to change this
		A_1u = D2 - B2 @ D3 @ B2.T
		A_1d = D2_inv - B1.T @ D1_inv @ B1
		A_1u_norm = (A_1u + torch.eye(n=A_1u.shape[0])) @ (torch.diag(1 / (D2diag + 1)))
		A_1d_norm = (D2 + torch.eye(n=D2.shape[0])) @ (A_1d + torch.eye(n=A_1d.shape[0]))
		L1 = A_1u_norm + A_1d_norm

		B2_sum = torch.sum(torch.abs(B2), 1)
		B2_sum_inv = 1 / (B2_sum + 1)
		D5inv = torch.diag(B2_sum_inv)

		A_2d = torch.eye(n=B2.shape[1]) + B2.T @ D5inv @ B2
		A_2d_norm = (2 * torch.eye(n=B2.shape[1])) @ (A_2d + torch.eye(n=A_2d.shape[0]))
		L2 = A_2d_norm

		B2D3 = B2 @ D3
		D2B1TD1inv = (1 / np.sqrt(2.)) * D2 @ B1.T @ D1_inv
		D1invB1 = (1 / np.sqrt(2.)) * D1_inv @ B1
		B2TD2inv = B2.T @ D5inv

		L0, L1, L2 = L0.to_sparse(), L1.to_sparse(), L2.to_sparse()
		B2D3, D2B1TD1inv, D1invB1, B2TD2inv = B2D3.to_sparse(), D2B1TD1inv.to_sparse(), \
																					D1invB1.to_sparse(), B2TD2inv.to_sparse()

		return SimplicialObject(X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv, label)


	def collate(self, data_list):
		X0, X1, X2 = [], [], []
		L0, L1, L2 = [], [], []
		# D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
		D1, D2, D3, D4 = [], [], [], []
		label = []

		x0_total, x1_total, x2_total = 0, 0, 0
		l0_total, l1_total, l2_total = 0, 0, 0
		d1_total, d2_total, d3_total, d4_total = 0, 0, 0, 0

		slices = {"X0": [0],
							"X1": [0],
							"X2": [0],
							"L0": [0],
							"L1": [0],
							"L2": [0],
							"B2D3" : [0],
							"D2B1TD1inv" : [0],
							"D1invB1" : [0],
							"B2TD2inv" : [0]}

		for data in data_list:
			x0, x1, x2 = data.X0, data.X1, data.X2
			l0, l1, l2 = data.L0, data.L1, data.L2
			d1, d2, d3, d4 = data.B2D3, data.D2B1TD1inv, data.D1invB1, data.B2TD2inv

			l = data.label
			x0_s, x1_s, x2_s = x0.shape[0], x1.shape[0], x2.shape[0]
			l0_s, l1_s, l2_s = l0.shape[1], l1.shape[1], l2.shape[1]
			d1_s, d2_s, d3_s, d4_s  = d1.shape[1], d2.shape[1], d3.shape[1], d4.shape[1]

			X0.append(x0)
			X1.append(x1)
			X2.append(x2)
			L0.append(l0)
			L1.append(l1)
			L2.append(l2)
			D1.append(d1)
			D2.append(d2)
			D3.append(d3)
			D4.append(d4)
			label.append(l)

			x0_total += x0_s
			x1_total += x1_s
			x2_total += x2_s
			l0_total += l0_s
			l1_total += l1_s
			l2_total += l2_s
			d1_total += d1_s
			d2_total += d2_s
			d3_total += d3_s
			d4_total += d4_s

			slices["X0"].append(x0_total)
			slices["X1"].append(x1_total)
			slices["X2"].append(x2_total)
			slices["L0"].append(l0_total)
			slices["L1"].append(l1_total)
			slices["L2"].append(l2_total)
			slices["B2D3"].append(d1_total)
			slices["D2B1TD1inv"].append(d2_total)
			slices["D1invB1"].append(d3_total)
			slices["B2TD2inv"].append(d4_total)

		X0 = torch.cat(X0, dim=0).to('cpu')
		X1 = torch.cat(X1, dim=0).to('cpu')
		X2 = torch.cat(X2, dim=0).to('cpu')
		L0 = torch.cat(L0, dim=-1).to('cpu')
		L1 = torch.cat(L1, dim=-1).to('cpu')
		L2 = torch.cat(L2, dim=-1).to('cpu')
		D1 = torch.cat(D1, dim=-1).to('cpu')
		D2 = torch.cat(D2, dim=-1).to('cpu')
		D3 = torch.cat(D3, dim=-1).to('cpu')
		D4 = torch.cat(D4, dim=-1).to('cpu')
		label = torch.cat(label, dim=-1).to('cpu')
		# D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
		data = SimplicialObject(X0, X1, X2, L0, L1, L2, D1, D2, D3, D4, label)

		return data, slices

	def get(self, data, slices, idx):
		x0_slice = slices["X0"][idx:idx + 2]
		x1_slice = slices["X1"][idx:idx + 2]
		x2_slice = slices["X2"][idx:idx + 2]
		l0_slice = slices["L0"][idx:idx + 2]
		l1_slice = slices["L1"][idx:idx + 2]
		l2_slice = slices["L2"][idx:idx + 2]
		d1_slice = slices["B2D3"][idx:idx + 2]
		d2_slice = slices["D2B1TD1inv"][idx:idx + 2]
		d3_slice = slices["D1invB1"][idx:idx + 2]
		d4_slice = slices["B2TD2inv"][idx:idx + 2]
		label_slice = [idx, idx + 1]

		X0 = data.X0[x0_slice[0]: x0_slice[1]]
		X1 = data.X1[x1_slice[0]: x1_slice[1]]
		X2 = data.X2[x2_slice[0]: x2_slice[1]]

		L0 = data.L0[:, l0_slice[0]: l0_slice[1]]
		L1 = data.L1[:, l1_slice[0]: l1_slice[1]]
		L2 = data.L2[:, l2_slice[0]: l2_slice[1]]

		# D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
		D1 = data.B2D3[:, d1_slice[0]: d1_slice[1]]
		D2 = data.D2B1TD1inv[:, d2_slice[0]: d2_slice[1]]
		D3 = data.D1invB1[:, d3_slice[0]: d3_slice[1]]
		D4 = data.B2TD2inv[:, d4_slice[0]: d4_slice[1]]

		label = data.label[label_slice[0]: label_slice[1]]

		return SimplicialObject(X0, X1, X2, L0, L1, L2, D1, D2, D3, D4, label)

	def batch(self, objectList):
		def unpack_SimplicialObject(SimplicialObject):
			X0, X1, X2 = SimplicialObject.X0, SimplicialObject.X1, SimplicialObject.X2
			L0, L1, L2 = SimplicialObject.L0, SimplicialObject.L1, SimplicialObject.L2

			L0_i, L0_v = L0[0:2], L0[2:3].squeeze()
			L1_i, L1_v = L1[0:2], L1[2:3].squeeze()
			L2_i, L2_v = L2[0:2], L2[2:3].squeeze()

			D1, D2, D3, D4 = SimplicialObject.B2D3, SimplicialObject.D2B1TD1inv, \
											 SimplicialObject.D1invB1, SimplicialObject.B2TD2inv

			D1_i, D1_v = D1[0:2], D1[2:3].squeeze()
			D2_i, D2_v = D2[0:2], D2[2:3].squeeze()
			D3_i, D3_v = D3[0:2], D3[2:3].squeeze()
			D4_i, D4_v = D4[0:2], D4[2:3].squeeze()
			# D1, D2, D3, D4 = B2D3, D2B1TD1inv, D1invB1, B2TD2inv
			label = SimplicialObject.label
			return [X0, X1, X2], [L0_i, L1_i, L2_i], [L0_v, L1_v, L2_v],\
				[D1_i, D2_i, D3_i, D4_i], [D1_v, D2_v, D3_v, D4_v], label

		unpacked_SimplicialObject = [unpack_SimplicialObject(g) for g in objectList]
		X, L_i, L_v, D_i, D_v, labels = [*zip(*unpacked_SimplicialObject)]
		X, L_i, L_v, D_i, D_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)], [*zip(*D_i)], [*zip(*D_v)]
		features_dct = batch_all_feature_and_lapacian_pair(X, L_i, L_v)
		D_i_batch, D_v_batch = [], []
		for i, v in zip(D_i, D_v):
			sizes_x = [1 + int(torch.max(matrix[0]).item()) for matrix in i]
			sizes_y = [1 + int(torch.max(matrix[1]).item()) for matrix in i]
			d_i_batch, d_v_batch = batch_sparse_matrix(i, v, sizes_x, sizes_y)
			D_i_batch.append(d_i_batch)
			D_v_batch.append(d_v_batch)
		features_dct['d_indices'] = D_i_batch
		features_dct['d_values'] = D_v_batch

		labels = torch.cat(labels, dim=0)
		return features_dct, labels

	def clean_feature_dct(self, feature_dct):
		feature_dct = convert_indices_and_values_to_sparse(feature_dct, 'lapacian_indices', 'lapacian_values', 'lapacian')
		return convert_indices_and_values_to_sparse(feature_dct, 'd_indices', 'd_values', 'others')
