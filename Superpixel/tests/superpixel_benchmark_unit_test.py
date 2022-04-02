import unittest
from Superpixel.ImageProcessor import ImageProcessor
import torch
from constants import TEST_MNIST_IMAGE_1, DEVICE
from utils import tensor_to_sparse
from Superpixel.EdgeFlow import PixelBasedEdgeFlow
from superpixel_benchmark import test
from torchvision import datasets
from Superpixel.SuperpixelDataset import SuperpixelSCDataset
from torch.utils.data import DataLoader
from models import superpixel_GCN, superpixel_GAT, superpixel_ESNN, superpixel_BSNN, superpixel_SAT, superpixel_SAN
from models.nn_utils import normalise, unpack_feature_dct_to_L_X_B
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
        for features_dct, train_labels in dataloader:
            features_dct = processor_type.clean_feature_dct(features_dct)
            features_dct = processor_type.repair(features_dct)
            features_dct = {key: convert_to_device(features_dct[key]) for key in features_dct}
            train_labels = train_labels.to(DEVICE)
            optimizer.zero_grad()
            prediction = NN(features_dct)
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
        processor = superpixel_GCN[0]()

        image = TEST_MNIST_IMAGE_1
        image = torch.tensor(image, dtype=torch.float, device=DEVICE)

        PI = ImageProcessor(sp_size, flow)
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
        features_test = torch.cat([features_test, features_test, features_test, features_test], dim=0)

        results = torch.allclose(features, features_test, atol=1e-5)

        self.assertTrue(results)

    def test_batching_gives_correct_result_2(self):
        sp_size = 100
        flow = PixelBasedEdgeFlow
        mulitplier = 5
        processor = superpixel_GCN[0]()

        image = TEST_MNIST_IMAGE_1
        image = torch.tensor(image, dtype=torch.float, device=DEVICE)

        PI = ImageProcessor(sp_size, flow)
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

        features_test = torch.cat([features_test, features_test, features_test, features_test], dim=0)

        results = torch.allclose(features, features_test, atol=1e-5)

        self.assertTrue(results)

    def test_gnn_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_GCN[1](5, 10).to(DEVICE)
        processor_type = superpixel_GCN[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 5000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        batched_predictions, _ = test(GNN, batched_dataset, processor_type)
        batched_predictions = torch.cat(batched_predictions, dim=0)

        individual_predictions, _ = test(GNN, individual_dataset, processor_type)
        individual_predictions = torch.cat(individual_predictions, dim=0)

        result = torch.allclose(individual_predictions, batched_predictions, atol=1e-5)

        self.assertTrue(result)

    def test_ebli_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_ESNN[1](5, 10, 15, 10).to(DEVICE)
        processor_type = superpixel_ESNN[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 5000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        batched_predictions, _ = test(GNN, batched_dataset, processor_type)
        batched_predictions = torch.cat(batched_predictions, dim=0)

        individual_predictions, _ = test(GNN, individual_dataset, processor_type)
        individual_predictions = torch.cat(individual_predictions, dim=0)

        result = torch.allclose(individual_predictions, batched_predictions, atol=1e-5)

        self.assertTrue(result)

    def test_bunch_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_BSNN[1](5, 10, 15, 10).to(DEVICE)
        processor_type = superpixel_BSNN[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 5000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        batched_predictions, _ = test(GNN, batched_dataset, processor_type)
        batched_predictions = torch.cat(batched_predictions, dim=0)

        individual_predictions, _ = test(GNN, individual_dataset, processor_type)
        individual_predictions = torch.cat(individual_predictions, dim=0)

        result = torch.allclose(individual_predictions, batched_predictions, atol=1e-5)

        self.assertTrue(result)

    def test_SAT_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_SAT[1](5, 10, 15, 10).to(DEVICE)
        processor_type = superpixel_SAT[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 5000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        _ = train(GNN, 1, batched_dataset, optimizer, criterion, processor_type)

        batched_predictions, _ = test(GNN, batched_dataset, processor_type)
        batched_predictions = torch.cat(batched_predictions, dim=0)

        individual_predictions, _ = test(GNN, individual_dataset, processor_type)
        individual_predictions = torch.cat(individual_predictions, dim=0)

        result = torch.allclose(individual_predictions, batched_predictions, atol=1e-5)

        self.assertTrue(result)


    def test_SAN_batching_gives_same_result_as_individual(self):
        batch_size = 8
        superpixel_size = 50
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow

        GNN = superpixel_SAN[1](5, 10, 15, 10).to(DEVICE)
        processor_type = superpixel_SAN[0]

        data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor, 5000,
                                   train=True)
        batched_dataset = DataLoader(data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                                     shuffle=False)
        individual_dataset = DataLoader(data, batch_size=1, collate_fn=processor_type.batch, num_workers=4,
                                        shuffle=False)

        optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        _ = train(GNN, 1, batched_dataset, optimizer, criterion, processor_type)

        batched_predictions, _ = test(GNN, batched_dataset, processor_type)
        batched_predictions = torch.cat(batched_predictions, dim=0)

        individual_predictions, _ = test(GNN, individual_dataset, processor_type)
        individual_predictions = torch.cat(individual_predictions, dim=0)

        result = torch.allclose(individual_predictions, batched_predictions, atol=1e-5)

        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
