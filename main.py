from dataset_processor.SuperpixelLoader import SimplicialComplexDataset
from dataset_processor.EdgeFlow import PixelBasedEdgeFlow, RandomBasedEdgeFlow, RAGBasedEdgeFlow
from torch.utils.data import DataLoader
from models.GNN.model import GCN, GAT
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.model_E import SNN_Ebli
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.model_B import SNN_Bunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from constants import DEVICE
import torch
from run_NN import test, train
from torchvision import datasets
import numpy as np
from datetime import timedelta

batch_size = 8
superpixel_size = 25
dataset = datasets.MNIST
# dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow
# edgeFlow = RAGBasedEdgeFlow
# edgeFlow = RandomBasedEdgeFlow

print(DEVICE)

# processor_type = GNNProcessor()
# processor_type = SNNEbliProcessor()
processor_type = SNNBunchProcessor()
output_size = 2
if __name__ == "__main__":

    GNN = SNN_Bunch(5, 10, 15, output_size).to(DEVICE)
    # GNN = SNN_Ebli(5, 10, 15, output_size).to(DEVICE)
    # GNN = GCN(5, output_size).to(DEVICE)
    # GNN = GAT(5, output_size).to(DEVICE)
    model_parameters = filter(lambda p: p.requires_grad, GNN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True)
    write_file_name = f"./results/{train_data.get_name()}"

    optimizer = torch.optim.Adam(GNN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    average_time, loss, final_loss, train_acc = train(GNN, 10, train_dataset, optimizer, criterion, processor_type)
    del train_data
    del train_dataset

    test_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True)

    _, test_acc = test(GNN, test_dataset, processor_type)

    with open(write_file_name + '.txt', 'w') as f:
        f.write(f"""Dataset: {dataset},\nModel: {GNN.__class__.__name__}\n\nparams={params}\n\n
    FINAL RESULTS\nTEST ACCURACY: {test_acc:.4f}\nTRAIN ACCURACY: {train_acc:.4f}\n\n
    Average Time Taken: {timedelta(seconds = average_time)}\nAverage Loss: {loss:.4f}\n\n\n""")

