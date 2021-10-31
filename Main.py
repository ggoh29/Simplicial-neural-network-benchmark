from dataset.ImageToDataset import ImageToSimplicialComplex
from torchvision import datasets
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
import torch
from multiprocessing import Pool
from models.GNN import GCN, GCN3
from dataset.MakeGraph import EdgeFlowSC, RAGSC
from models.SNN import SNN
from constants import DEVICE

train_data = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)

# train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
# test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 16

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

superpixel_size = 75

if __name__ == "__main__":

    I2G = ImageToSimplicialComplex(superpixel_size, EdgeFlowSC, simplicial_complex_size=2)

    GNN = SNN(5, 10, 15, 10).to(DEVICE)
    # GNN = GCN().to(DEVICE)
    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    GNN.train()

    with Pool(8) as p:
        for epoch in range(25):
            train_acc = 0
            i = 0
            train_running_loss = 0
            for train_features, train_labels in train_dataloader:
                train_features.to(DEVICE), train_labels.to(DEVICE)
                optimizer.zero_grad()
                prediction = I2G.process_batch_and_feed_to_NN(GNN, train_features, p)
                loss = criterion(prediction, train_labels)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.detach().item()
                train_acc += (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
                i += 1

            print(f"Epoch {epoch} | Train running loss {train_running_loss / i} | Loss {loss} | Train accuracy {train_acc / i}")

    GNN.eval()

    test_acc = 0
    i = 0
    with Pool(8) as p:
        with torch.no_grad():
            for test_features, test_labels in test_dataloader:
                test_features.to(DEVICE), test_labels.to(DEVICE)
                prediction = I2G.process_batch_and_feed_to_NN(GNN, test_features, p)
                test_acc += (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
                i += 1

    print(test_acc/i)
