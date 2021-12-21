import unittest
from dataset_processor.SuperpixelDataset.ImageProcessor import ProcessImage
import torch
from constants import TEST_MNIST_IMAGE_1, DEVICE
# DEVICE = torch.device('cpu')
from utils import tensor_to_sparse
from dataset_processor.SuperpixelDataset.EdgeFlow import PixelBasedEdgeFlow
from run_NN import test, train
from torchvision import datasets
from dataset_processor.SuperpixelDataset.SuperpixelLoader import SimplicialComplexDataset
from torch.utils.data import DataLoader
from models.GNN.model import GCN
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.model_E import SNN_Ebli
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.model_B import SNN_Bunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from models.nn_utils import normalise, unpack_feature_dct_to_L_X_B
from models.SAT.model import SAT
from models.SAT.SATProcessor import SATProcessor

class MyTestCase(unittest.TestCase):

	def test_batching_gives_correct_result_1(self):
		sp_size = 100
		flow = PixelBasedEdgeFlow
		processor = GNNProcessor()

		image = TEST_MNIST_IMAGE_1
		image = torch.tensor(image, dtype=torch.float, device=DEVICE)

		PI = ProcessImage(sp_size, flow)
		scData = PI.image_to_features((image, 0))

		batch = [scData, scData, scData, scData]
		batch = [processor.process(i) for i in batch]

		features_dct, _ = processor.batch(batch)
		features_dct = processor.clean_feature_dct(features_dct)
		lapacian, features, _ = unpack_feature_dct_to_L_X_B(features_dct)
		lapacian, features = lapacian[0], features[0]

		features = torch.sparse.mm(lapacian, features)

		features_test = scData.X0
		b1 = tensor_to_sparse(scData.b1)
		lapacian_test = normalise(torch.sparse.mm(b1, b1.t())).to_dense()

		features_test = torch.sparse.mm(lapacian_test, features_test)
		features_test = torch.cat([features_test, features_test, features_test, features_test], dim = 0)

		results = torch.allclose(features, features_test, atol = 1e-5)

		self.assertTrue(results)

	def test_batching_gives_correct_result_2(self):
		sp_size = 100
		flow = PixelBasedEdgeFlow
		mulitplier = 5
		processor = GNNProcessor()

		image = TEST_MNIST_IMAGE_1
		image = torch.tensor(image, dtype=torch.float, device=DEVICE)

		PI = ProcessImage(sp_size, flow)
		scData = PI.image_to_features((image, 0))

		batch = [scData, scData, scData, scData]
		batch = [processor.process(i) for i in batch]

		features_dct, _ = processor.batch(batch)
		features_dct = processor.clean_feature_dct(features_dct)
		lapacian, features, _ = unpack_feature_dct_to_L_X_B(features_dct)
		lapacian, features = lapacian[0], features[0]

		b1 = tensor_to_sparse(scData.b1)
		lapacian_test = normalise(torch.sparse.mm(b1, b1.t())).to_dense()

		# Rounding to avoid floating point errors

		features_test = scData.X0

		for _ in range(mulitplier):
			features_test = torch.sparse.mm(lapacian_test, features_test)
			features = torch.sparse.mm(lapacian, features)

		features_test = torch.cat([features_test, features_test, features_test, features_test], dim = 0)

		results = torch.allclose(features, features_test, atol = 1e-5)

		self.assertTrue(results)

	def test_gnn_batching_gives_same_result_as_individual(self):
		batch_size = 8
		superpixel_size = 50
		dataset = datasets.MNIST
		edgeFlow = PixelBasedEdgeFlow

		GNN = GCN(5, 4).to(DEVICE)
		processor_type = GNNProcessor()

		data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
		batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)
		individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)

		batched_predictions, _ = test(GNN, batched_dataset, processor_type)
		batched_predictions = torch.cat(batched_predictions, dim = 0)

		individual_predictions, _ = test(GNN, individual_dataset, processor_type)
		individual_predictions = torch.cat(individual_predictions, dim = 0)

		result = torch.allclose(individual_predictions, batched_predictions, atol = 1e-5)

		self.assertTrue(result)


	def test_ebli_batching_gives_same_result_as_individual(self):
		batch_size = 8
		superpixel_size = 50
		dataset = datasets.MNIST
		edgeFlow = PixelBasedEdgeFlow

		GNN = SNN_Ebli(5, 10, 15, 10).to(DEVICE)
		processor_type = SNNEbliProcessor()

		data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
		batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)
		individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)

		batched_predictions, _ = test(GNN, batched_dataset, processor_type)
		batched_predictions = torch.cat(batched_predictions, dim = 0)

		individual_predictions, _ = test(GNN, individual_dataset, processor_type)
		individual_predictions = torch.cat(individual_predictions, dim = 0)

		result = torch.allclose(individual_predictions, batched_predictions, atol = 1e-5)

		self.assertTrue(result)

	def test_bunch_batching_gives_same_result_as_individual(self):
		batch_size = 8
		superpixel_size = 50
		dataset = datasets.MNIST
		edgeFlow = PixelBasedEdgeFlow

		GNN = SNN_Bunch(5, 10, 15, 10).to(DEVICE)
		processor_type = SNNBunchProcessor()

		data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
		batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)
		individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)

		batched_predictions, _ = test(GNN, batched_dataset, processor_type)
		batched_predictions = torch.cat(batched_predictions, dim = 0)

		individual_predictions, _ = test(GNN, individual_dataset, processor_type)
		individual_predictions = torch.cat(individual_predictions, dim = 0)

		result = torch.allclose(individual_predictions, batched_predictions, atol = 1e-5)

		self.assertTrue(result)


	def test_SAT_batching_gives_same_result_as_individual(self):
		batch_size = 8
		superpixel_size = 50
		dataset = datasets.MNIST
		edgeFlow = PixelBasedEdgeFlow

		GNN = SAT(5, 10, 15, 10).to(DEVICE)
		processor_type = SATProcessor()

		data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
		batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)
		individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
															 shuffle=False)

		optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
		criterion = torch.nn.CrossEntropyLoss()

		_ = train(GNN, 1, batched_dataset, optimizer, criterion, processor_type)

		batched_predictions, _ = test(GNN, batched_dataset, processor_type)
		batched_predictions = torch.cat(batched_predictions, dim = 0)

		individual_predictions, _ = test(GNN, individual_dataset, processor_type)
		individual_predictions = torch.cat(individual_predictions, dim = 0)

		result = torch.allclose(individual_predictions, batched_predictions, atol = 1e-5)

		self.assertTrue(result)




if __name__ == '__main__':
	unittest.main()
