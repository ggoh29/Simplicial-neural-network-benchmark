from dataset_processor.SuperpixelLoader import DatasetType, SimplicialComplexDataset
from dataset_processor.EdgeFlow import PixelBasedEdgeFlowSC, RAGBasedEdgeFlow
from torch.utils.data import DataLoader
from models.SNN import SNN
from models.GNN import GCN
from constants import DEVICE
import torch
from run_NN import test, train

batch_size = 8
superpixel_size = 75
dataset = DatasetType.MNIST
#dataset_processor = DatasetType.CIFAR10

if __name__ == "__main__":
    train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, PixelBasedEdgeFlowSC, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.batch, num_workers=2, shuffle=True)
    test_data = SimplicialComplexDataset('./data', dataset, superpixel_size, PixelBasedEdgeFlowSC, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=test_data.batch, num_workers=2,
                              shuffle=True)

    GNN = SNN(5, 10, 15, 10).to(DEVICE)
    # GNN = GCN(5, 10).to(DEVICE)
    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(GNN, 100, train_dataset, optimizer, criterion)
    test(GNN, test_dataset)

