from DatasetProcessing.ImageToDataset import ImageToGraph
from torchvision import datasets
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
import torch
from skimage.future import graph
from skimage.segmentation import slic
from skimage.measure import regionprops
import numpy as np
from multiprocessing import Pool
from models.GNN import GCN
from models.SNN import SNN

train_data = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)

batch_size = 1

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

superpixel_size = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unpack_node(node):
    c = 'centroid'
    mc = 'mean color'
    # pc = 'pixel count'
    # return [node[mc][0], node[mc][1], node[mc][2]]
    return [node[c][0], node[c][1], node[mc][0], node[mc][1], node[mc][2]]


def image_to_superpixel(img, superpixel_size=superpixel_size):
    # converting image to superpixel
    img = (img.double().numpy())[0]
    superpixel = slic(img, n_segments=superpixel_size, compactness=0.75, start_label = 1)

    rag = graph.rag_mean_color(img, superpixel)
    regions = regionprops(superpixel)

    for region in regions:
        rag._node[region['label']]['centroid'] = region['centroid']

    return superpixel, rag, img


def image_to_feature_and_adjacency(img, superpixel_size=superpixel_size):
    superpixel, rag, _ = image_to_superpixel(img, superpixel_size)
    F = torch.tensor([unpack_node(rag._node[node]) for node in rag._node],
                     dtype=torch.float, device=device)

    # undirected edge list
    A1 = torch.tensor([[a[0] - 1, a[1] - 1] for a in rag.edges()], device=device).T
    A2 = torch.tensor([[a[1] - 1, a[0] - 1] for a in rag.edges()], device=device).T
    A = torch.cat((A1, A2), 1)

    return F, A, len(rag._node)


def process_batch_and_feed_to_NN(NN, image_list, p):
    # generate feature, edge and batch list from images
    features_and_edge_list = p.map(image_to_feature_and_adjacency, image_list)
    features, edges, node_size = [*zip(*features_and_edge_list)]

    features = torch.cat(features, dim=0)

    edges, node_size = list(edges), list(node_size)
    node_size.insert(0, 0)
    mx = 0
    edge_len = len(edges)
    for i in range(edge_len):
        mx += node_size[i]
        edges[i] += mx
    node_size.pop(0)

    edges = torch.cat(edges, dim=1)
    batch = [[i for _ in range(node_size[i])] for i in range(edge_len)]
    batch = torch.tensor([i for sublist in batch for i in sublist], device=device)
    return NN(features, edges, batch)


if __name__ == "__main__":

    I2G = ImageToGraph(superpixel_size)

    GNN = SNN(5,10,15,10).to(device)
    optimizer_GNN = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    GNN.train()

    with Pool(8) as p:
        for epoch in range(1):
            train_acc = 0
            i = 0
            train_running_loss = 0
            for train_features, train_labels in train_dataloader:
                train_features.to(device), train_labels.to(device)
                optimizer_GNN.zero_grad()
                prediction = I2G.process_batch_and_feed_to_NN(GNN, train_features, p)
                loss = criterion(prediction, train_labels)
                loss.backward()
                optimizer_GNN.step()

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
                test_features.to(device), test_labels.to(device)
                prediction = process_batch_and_feed_to_NN(GNN, test_features, p)
                test_acc += (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
                i += 1

    print(test_acc/i)