from DatasetProcessing.ImageToDataset import ImageToGraph
from torchvision import datasets
import torchvision.transforms as transforms
from torch_geometric.data import DataLoader

data = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(data, batch_size=1, shuffle=True)


if __name__ == "__main__":
    i = iter(train_dataloader)
    next(i)
    train_features, train_labels = next(i)
    I2G = ImageToGraph(50)
    print(train_features[0])
    I2G.getFeaturesAndLapacians(train_features[0])
