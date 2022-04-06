import unittest
from Superpixel.ImageProcessor import ImageProcessor
import torch
from constants import TEST_MNIST_IMAGE_1, DEVICE, TEST_CIFAR10_IMAGE_1
from utils import tensor_to_sparse
from Superpixel.EdgeFlow import PixelBasedEdgeFlow
from superpixel_benchmark import test
from torchvision import datasets
from Superpixel.SuperpixelDataset import SuperpixelSCDataset
from torch.utils.data import DataLoader
from models import superpixel_GCN, superpixel_GAT, superpixel_ESNN, superpixel_BSNN, superpixel_SAT, superpixel_SAN, \
    test_ESNN, test_SAT, test_SAN, test_BSNN
from models.nn_utils import normalise, to_sparse_coo
from models.SAT.SATProcessor import SATProcessor
import time


def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]


def train(NN, epoch_size, dataloader, optimizer, criterion, processor_type):
    NN.train()
    train_running_loss = 0
    t = 0
    for epoch in range(epoch_size):
        t1 = time.perf_counter()
        epoch_train_running_loss = 0
        train_acc = 0
        i = 0
        for simplicialComplex, train_labels in dataloader:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            simplicialComplex.to_device()
            train_labels = train_labels.to(DEVICE)
            optimizer.zero_grad()
            prediction = NN(simplicialComplex)
            loss = criterion(prediction, train_labels)
            loss.backward()
            optimizer.step()
            epoch_train_running_loss += loss.detach().item()
            train_acc += (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
            i += 1
        t2 = time.perf_counter()
        t = (t * epoch + (t2 - t1)) / (epoch + 1)
        epoch_train_running_loss /= i
        train_running_loss = (train_running_loss * epoch + epoch_train_running_loss) / (epoch + 1)
        print(
            f"Epoch {epoch} | Train running loss {train_running_loss} "
            f"| Loss {epoch_train_running_loss} | Train accuracy {train_acc / i}")
        epoch_loss = epoch_train_running_loss
        acc = train_acc / i
    return t, train_running_loss, epoch_loss, acc


class MyTestCase(unittest.TestCase):

    def test_batching_gives_correct_result_1(self):
        sp_size = 100
        flow = PixelBasedEdgeFlow
        processor = superpixel_GCN[0]

        image = TEST_MNIST_IMAGE_1
        image = torch.tensor(image, dtype=torch.float)

        PI = ImageProcessor(sp_size, flow)
        scData = PI.image_to_features((image, 0))

        batch = [scData, scData, scData, scData]
        batch = [processor.process(i) for i in batch]

        simplicialComplex, _ = processor.batch(batch)
        simplicialComplex = processor.clean_features(simplicialComplex)
        laplacian = simplicialComplex.L0
        features = simplicialComplex.X0

        features = torch.sparse.mm(laplacian, features)

        features_test = scData.X0
        b1 = tensor_to_sparse(scData.b1)
        lapacian_test = torch.sparse.mm(b1, b1.t()).to_dense()

        features_test = torch.sparse.mm(lapacian_test, features_test)
        features_test = torch.cat([features_test, features_test, features_test, features_test], dim=0)

        results = torch.allclose(features, features_test, atol=1e-5)

        self.assertTrue(results)

    def test_batching_gives_correct_result_2(self):
        sp_size = 100
        flow = PixelBasedEdgeFlow
        mulitplier = 5
        processor = superpixel_GCN[0]

        image = TEST_CIFAR10_IMAGE_1
        image = torch.tensor(image, dtype=torch.float)

        PI = ImageProcessor(sp_size, flow)
        scData = PI.image_to_features((image, 0))

        batch = [scData, scData, scData, scData]
        batch = [processor.process(i) for i in batch]

        simplicialComplex, _ = processor.batch(batch)
        simplicialComplex = processor.clean_features(simplicialComplex)
        laplacian = simplicialComplex.L0
        features = simplicialComplex.X0

        b1 = tensor_to_sparse(scData.b1)
        lapacian_test = torch.sparse.mm(b1, b1.t()).to_dense()

        features_test = scData.X0

        for _ in range(mulitplier):
            features_test = torch.sparse.mm(lapacian_test, features_test)
            features = torch.sparse.mm(laplacian, features)

        features_test = torch.cat([features_test, features_test, features_test, features_test], dim=0)

        results = torch.allclose(features, features_test, atol=1e-5)

        self.assertTrue(results)

    def test_gnn_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_GCN[1](5, 10)
        processor_type = superpixel_GCN[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 1000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)
        GNN.eval()
        batched1 = []
        for simplicialComplex, test_labels in batched_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            b1 = GNN(simplicialComplex)
            batched1.append(b1)

        individual1 = []
        for simplicialComplex, test_labels in individual_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            i1 = GNN(simplicialComplex)
            individual1.append(i1)
        result1 = torch.allclose(torch.cat(batched1, dim=0), torch.cat(individual1, dim=0), atol=1e-5)
        self.assertTrue(result1)

    def test_ebli_batching_gives_same_result_as_individual(self):
        batch_size = 16
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_ESNN[1](5, 10, 15, 10).to(DEVICE)
        processor_type = superpixel_ESNN[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 1000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)
        batched1 = []
        for simplicialComplex, test_labels in batched_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            b1 = GNN(simplicialComplex)
            batched1.append(b1)

        individual1 = []
        for simplicialComplex, test_labels in individual_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            i1 = GNN(simplicialComplex)
            individual1.append(i1)
        result1 = torch.allclose(torch.cat(batched1, dim=0), torch.cat(individual1, dim=0), atol=1e-4)
        self.assertTrue(result1)

    def test_bunch_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = test_BSNN[1](5, 10, 15, 10).to(DEVICE)
        processor_type = superpixel_BSNN[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 1000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        batched1 = []
        for simplicialComplex, test_labels in batched_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            b1 = GNN(simplicialComplex)
            batched1.append(b1)

        individual1 = []
        for simplicialComplex, test_labels in individual_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            i1 = GNN(simplicialComplex)
            individual1.append(i1)
        result1 = torch.allclose(torch.cat(batched1, dim=0), torch.cat(individual1, dim=0), atol=1e-4)
        self.assertTrue(result1)


    def test_SAT_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_SAT[1](5, 10, 15, 10)
        processor_type = superpixel_SAT[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 1000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        batched1 = []
        for simplicialComplex, test_labels in batched_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            b1 = GNN(simplicialComplex)
            batched1.append(b1)

        individual1 = []
        for simplicialComplex, test_labels in individual_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            i1 = GNN(simplicialComplex)
            individual1.append(i1)
        result1 = torch.allclose(torch.cat(batched1, dim=0), torch.cat(individual1, dim=0), atol=1e-4)
        self.assertTrue(result1)

    def test_SAN_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_SAN[1](5, 10, 15, 10)
        processor_type = superpixel_SAN[0]

        data = SuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 1000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        batched1 = []
        for simplicialComplex, test_labels in batched_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            b1 = GNN(simplicialComplex)
            batched1.append(b1)

        individual1 = []
        for simplicialComplex, test_labels in individual_dataset:
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            i1 = GNN(simplicialComplex)
            individual1.append(i1)
        result1 = torch.allclose(torch.cat(batched1, dim=0), torch.cat(individual1, dim=0), atol=1e-4)
        self.assertTrue(result1)


if __name__ == '__main__':
    unittest.main()
