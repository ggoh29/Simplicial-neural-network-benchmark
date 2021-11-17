from dataset_processor.SuperpixelLoader import SimplicialComplexDataset
from dataset_processor.EdgeFlow import PixelBasedEdgeFlow
from torch.utils.data import DataLoader
from models.GNN.model import GCN
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Stefanie.model_S import SNN
from models.SNN_Stefanie.SNNStefProcessor import SNNStefProcessor
from constants import DEVICE
import torch
from run_NN import test, train
from torchvision import datasets

batch_size = 8
superpixel_size = 50
dataset = datasets.MNIST
# dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow
# edgeFlow = RAGBasedEdgeFlow


processor_type = GNNProcessor()
# processor_type = SNNStefProcessor()

if __name__ == "__main__":

    # GNN = SNN(5, 10, 15, 4).to(DEVICE)
    GNN = GCN(5, 4).to(DEVICE)
    # GNN = GCN2(5, 10).to(DEVICE)

    train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True)
    test_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True)

    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(GNN, 100, train_dataset, optimizer, criterion, processor_type)
    test(GNN, test_dataset, processor_type)

