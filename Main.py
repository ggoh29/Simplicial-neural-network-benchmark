import torch
from models.GNN import GCN, GCN3
from dataset.SuperpixelLoader import SuperPixelLoader
from dataset.MakeGraph import EdgeFlowSC, RAGSC
from models.SNN import SNN
from constants import DEVICE
from run_tests import train, test
from multiprocessing import Pool


batch_size = 16
superpixel_size = 100

if __name__ == "__main__":

    train_dataset = SuperPixelLoader("CIFAR10", superpixel_size, EdgeFlowSC, True, batch_size, 4)
    test_dataset = SuperPixelLoader("CIFAR10", superpixel_size, EdgeFlowSC, False, batch_size, 4)

    GNN = SNN(5, 10, 15, 10).to(DEVICE)
    # GNN = GCN().to(DEVICE)
    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(GNN, 25, train_dataset, optimizer, criterion)
    test(GNN, test_dataset)

