import unittest
from dataset_processor.ImageProcessor import ProcessImage
import torch
from constants import TEST_MNIST_IMAGE_1, DEVICE
from utils import tensor_to_dense
import numpy as np
from dataset_processor.EdgeFlow import PixelBasedEdgeFlow, RAGBasedEdgeFlow
from run_NN import test, train
from torchvision import datasets
from dataset_processor.SuperpixelLoader import SimplicialComplexDataset
from torch.utils.data import DataLoader
from models.SNN import SNN
from models.nn_utils import collated_data_to_batch, convert_indices_and_values_to_sparse, unpack_feature_dct_to_L_X_B

class MyTestCase(unittest.TestCase):

	def test_batching_gives_correct_result_1(self):
		sp_size = 100
		flow = PixelBasedEdgeFlow

		image = TEST_MNIST_IMAGE_1
		image = torch.tensor(image, dtype=torch.float, device=DEVICE)

		PI = ProcessImage(sp_size, flow)
		scData = PI.image_to_features((image, 0))

		batch = [scData, scData, scData, scData]

		features_dct, _  = collated_data_to_batch(batch)
		features_dct = convert_indices_and_values_to_sparse(features_dct)
		features, lapacian, _ = unpack_feature_dct_to_L_X_B(features_dct)

		features = torch.sparse.mm(lapacian, features)

		features_test = scData.X0
		sigma1 = tensor_to_dense(scData.sigma1)
		lapacian_test = torch.sparse.mm(sigma1, sigma1.t()).to_dense()

		features_test = torch.sparse.mm(lapacian_test, features_test)
		features_test = torch.cat([features_test, features_test, features_test, features_test], dim = 0)

		# Convert to np so I can change to 4dp.
		results = np.around(features - features_test, decimals=4) == 0

		self.assertTrue(results.all().item())

	def test_batching_gives_correct_result_2(self):
		sp_size = 100
		flow = PixelBasedEdgeFlow
		mulitplier = 5

		image = TEST_MNIST_IMAGE_1
		image = torch.tensor(image, dtype=torch.float, device=DEVICE)

		PI = ProcessImage(sp_size, flow)
		scData = PI.image_to_features((image, 0))

		batch = [scData, scData, scData, scData]

		features_dct, _ = collated_data_to_batch(batch)
		features_dct = convert_indices_and_values_to_sparse(features_dct)
		features, lapacian, _ = unpack_feature_dct_to_L_X_B(features_dct)

		sigma1 = tensor_to_dense(scData.sigma1)
		lapacian_test = torch.sparse.mm(sigma1, sigma1.t()).to_dense()

		# Rounding to avoid floating point errors
		features = features[0].round()
		features_test = scData.X0.round()

		for _ in range(mulitplier):
			features_test = torch.sparse.mm(lapacian_test, features_test)
			features = torch.sparse.mm(lapacian, features)

		features_test = torch.cat([features_test, features_test, features_test, features_test], dim = 0)

		# Convert to np array so I can change to 4dp.
		results = np.around(features - features_test, decimals=4) == 0

		self.assertTrue(results.all().item())


	def test_batching_gives_same_result_as_individual(self):
		batch_size = 8
		superpixel_size = 50
		dataset = datasets.MNIST
		edgeFlow = RAGBasedEdgeFlow
		GNN = SNN(5, 10, 15, 4).to(DEVICE)
		# GNN = GCN(5, 4).to(DEVICE)

		data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, train=True)
		batched_data = DataLoader(data, batch_size=8, collate_fn=GNN.batch, num_workers=1, shuffle=False)
		individual_data = DataLoader(data, batch_size=1, collate_fn=GNN.batch, num_workers=1, shuffle=False)

		optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
		criterion = torch.nn.CrossEntropyLoss()

		train(GNN, 1, individual_data, optimizer, criterion)

		batched_predictions = test(GNN, batched_data)
		batched_predictions = torch.cat(batched_predictions, dim = 0)

		individual_predictions = test(GNN, individual_data)
		individual_predictions = torch.cat(individual_predictions, dim = 0)

		result = torch.allclose(individual_predictions, batched_predictions, atol = 1e-5)

		self.assertTrue(result)




if __name__ == '__main__':
	unittest.main()
