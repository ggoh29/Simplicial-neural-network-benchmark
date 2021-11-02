import torch
from models.GNN import GCN, GCN3
from dataset.SuperpixelLoader import SuperPixelLoader, DatasetType
from dataset.MakeGraph import EdgeFlowSC, RAGSC
from models.SNN import SNN
from constants import DEVICE
from run_NN import train, test
from multiprocessing import Pool


batch_size = 8
superpixel_size = 75

if __name__ == "__main__":

    dataset = DatasetType.MNIST

    train_dataset = SuperPixelLoader(dataset, superpixel_size, EdgeFlowSC, True, batch_size, 4, simplicial_complex_size=2)
    test_dataset = SuperPixelLoader(dataset, superpixel_size, EdgeFlowSC, False, batch_size, 4, simplicial_complex_size=2)

    GNN = SNN(5, 10, 15, 4).to(DEVICE)
    # GNN = GCN(5, 4).to(DEVICE)
    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(GNN, 25, train_dataset, optimizer, criterion)
    test(GNN, test_dataset)

