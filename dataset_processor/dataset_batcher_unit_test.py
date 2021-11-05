import unittest
from dataset_processor.ImageProcessor import ProcessImage
import torch
from dataset_processor.DatasetBatcher import DatasetBatcher
from dataset_processor.EdgeFlow import PixelBasedEdgeFlowSC
from constants import TEST_MNIST_IMAGE_1, DEVICE
from utils import tensor_to_dense
import numpy as np

class MyTestCase(unittest.TestCase):

	def test_batching_gives_correct_result_1(self):
		sp_size = 100
		flow = PixelBasedEdgeFlowSC

		image = TEST_MNIST_IMAGE_1
		image = torch.tensor(image, dtype=torch.float, device=DEVICE)

		PI = ProcessImage(sp_size, flow)
		scData = PI.image_to_features((image, 0))

		batch = [scData, scData, scData, scData]

		db = DatasetBatcher(simplicial_complex_size=0)
		(features, l_i, v_i, _) , _ = db.collated_data_to_batch(batch)
		lapacian = torch.sparse_coo_tensor(l_i[0], v_i[0])
		features = features[0]

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
		flow = PixelBasedEdgeFlowSC
		mulitplier = 5

		image = TEST_MNIST_IMAGE_1
		image = torch.tensor(image, dtype=torch.float, device=DEVICE)

		PI = ProcessImage(sp_size, flow)
		scData = PI.image_to_features((image, 0))

		batch = [scData, scData, scData, scData]

		db = DatasetBatcher(simplicial_complex_size=0)
		(features, l_i, v_i, _) , _ = db.collated_data_to_batch(batch)
		lapacian = torch.sparse_coo_tensor(l_i[0], v_i[0])

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

if __name__ == '__main__':
	unittest.main()
