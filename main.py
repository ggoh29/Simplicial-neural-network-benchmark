from dataset_processor.SuperpixelLoader import SimplicialComplexDataset
from dataset_processor.EdgeFlow import PixelBasedEdgeFlow
from torch.utils.data import DataLoader
from models.GNN.model import GCN, GAT
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Stefanie.model_S import SNN_Stef
from models.SNN_Stefanie.SNNStefProcessor import SNNStefProcessor
from models.SNN_Bunch.model_B import SNN_Bunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from constants import DEVICE
import torch
from run_NN import test, train
from torchvision import datasets
import numpy as np

batch_size = 8
superpixel_size = 75
# dataset = datasets.MNIST
dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow
# edgeFlow = RAGBasedEdgeFlow

# processor_type = GNNProcessor()
# processor_type = SNNStefProcessor()
processor_type = SNNBunchProcessor()
output_size = 10
if __name__ == "__main__":

    GNN = SNN_Bunch(5, 10, 15, output_size).to(DEVICE)
    # GNN = SNN_Stef(5, 10, 15, output_size).to(DEVICE)
    # GNN = GCN(5, output_size).to(DEVICE)
    # GNN = GAT(5, output_size).to(DEVICE)
    model_parameters = filter(lambda p: p.requires_grad, GNN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True)
    test_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True)

    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(GNN, 100, train_dataset, optimizer, criterion, processor_type)
    test(GNN, test_dataset, processor_type)
