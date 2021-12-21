from dataset_processor.SuperpixelDataset.SuperpixelLoader import SimplicialComplexDataset
from dataset_processor.SuperpixelDataset.EdgeFlow import PixelBasedEdgeFlow
from torch.utils.data import DataLoader
from models.GNN.model import GCN, GAT
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.model_E import SNN_Ebli
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.model_B import SNN_Bunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from models.SAT.model import SAT
from models.SAT.SATProcessor import SATProcessor
from constants import DEVICE
import torch
from run_NN import test, train
from torchvision import datasets
import numpy as np
from datetime import timedelta

batch_size = 8
superpixel_size = 75
dataset = datasets.MNIST
# dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow
# edgeFlow = RAGBasedEdgeFlow
# edgeFlow = RandomBasedEdgeFlow

output_size = 10

Ebli_nn = [SNNEbliProcessor(), SNN_Ebli(5, 10, 15, output_size)]
Bunch_nn = [SNNBunchProcessor(), SNN_Bunch(5, 10, 15, output_size)]
sat_nn = [SATProcessor(), SAT(5, 10, 15, output_size)]
gnn = [GNNProcessor(), GCN(5, output_size)]
gat = [GNNProcessor(), GAT(5, output_size)]


def run(processor_type, NN, output_suffix):
    model_parameters = filter(lambda p: p.requires_grad, NN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True, pin_memory=True)
    write_file_name = f"./results/{train_data.get_name()}_{NN.__class__.__name__}_{output_suffix}"

    test_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4, shuffle=True, pin_memory=True)


    optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    average_time, loss, final_loss, train_acc = train(NN, 100, train_dataset, optimizer, criterion, processor_type)

    _, test_acc = test(NN, test_dataset, processor_type)

    s = f"""Dataset: {dataset},\nModel: {NN.__class__.__name__}\n\nparams={params}\n\n
    FINAL RESULTS\nTEST ACCURACY: {test_acc:.4f}\nTRAIN ACCURACY: {train_acc:.4f}\n\n
    Average Time Taken: {timedelta(seconds = average_time)}\nAverage Loss: {loss:.4f}\n\n\n"""

    print(s)

    with open(write_file_name + '.txt', 'w') as f:
        f.write(s)



if __name__ == "__main__":
    # NN_list = [gnn, Ebli_nn, Bunch_nn, sat_nn]
    NN_list = [sat_nn]
    for output_suffix in range(5):
        for processor_type, NN in NN_list:
            run(processor_type, NN.to(DEVICE), output_suffix)



