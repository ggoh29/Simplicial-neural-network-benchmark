from dataset_processor.SuperpixelLoader import SimplicialComplexDataset
from dataset_processor.EdgeFlow import PixelBasedEdgeFlow, RAGBasedEdgeFlow
from torch.utils.data import DataLoader
from models.SNN import SNN
from models.GNN import GCN, GCN2, GCN3
from constants import DEVICE
import torch
from run_NN import test, train
from torchvision import datasets

batch_size = 8
superpixel_size = 25
dataset = datasets.MNIST
# dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow
edgeFlow = RAGBasedEdgeFlow

if __name__ == "__main__":
    train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow , complex_size=2, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.batch, num_workers=4, shuffle=True)
    test_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow ,complex_size=2, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=test_data.batch, num_workers=4,
                              shuffle=True)

    GNN = SNN(5, 10, 15, 2).to(DEVICE)

    # GNN = GCN3(5, 10).to(DEVICE)
    # GNN = GCN2(5, 10, batch_size).to(DEVICE)
    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(GNN, 100, train_dataset, optimizer, criterion)
    test(GNN, test_dataset)

