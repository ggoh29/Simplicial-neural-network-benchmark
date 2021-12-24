from dataset_processor.SuperpixelDataset.SuperpixelLoader import SuperpixelSCDataset
from dataset_processor.SuperpixelDataset.EdgeFlow import PixelBasedEdgeFlow, RandomBasedEdgeFlow
from torch.utils.data import DataLoader
from constants import DEVICE
from models.all_models import Ebli_nn, Bunch_nn, sat_nn, gnn, gat
import torch
from torchvision import datasets
import numpy as np
from datetime import timedelta
import time

batch_size = 8
superpixel_size = 75
dataset = datasets.MNIST
# dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow
# edgeFlow = RandomBasedEdgeFlow

output_size = 10

def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]


def train(NN, epoch_size, dataloader, optimizer, criterion, processor_type):
    NN.train()
    train_running_loss = 0
    t = 0
    for epoch in range(epoch_size):
        t1 = time.perf_counter()
        epoch_train_running_loss = 0
        train_acc = 0
        i = 0
        for features_dct, train_labels in dataloader:
            features_dct = processor_type.clean_feature_dct(features_dct)
            features_dct = {key: convert_to_device(features_dct[key]) for key in features_dct}
            train_labels = train_labels.to(DEVICE)
            optimizer.zero_grad()
            prediction = NN(features_dct)
            loss = criterion(prediction, train_labels)
            loss.backward()
            optimizer.step()
            epoch_train_running_loss += loss.detach().item()
            train_acc += (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
            i += 1
        t2 = time.perf_counter()
        t = (t * epoch + (t2 - t1)) / (epoch + 1)
        epoch_train_running_loss /= i
        train_running_loss = (train_running_loss * epoch + epoch_train_running_loss) / (epoch + 1)
        print(
            f"Epoch {epoch} | Train running loss {train_running_loss} "
            f"| Loss {epoch_train_running_loss} | Train accuracy {train_acc / i}")
        epoch_loss = epoch_train_running_loss
        acc = train_acc / i
    return t, train_running_loss, epoch_loss, acc


def test(NN, dataloader, processor_type):
    NN.eval()

    test_acc = 0
    i = 0
    predictions = []
    with torch.no_grad():
        for features_dct, test_labels in dataloader:
            features_dct = processor_type.clean_feature_dct(features_dct)
            features_dct = {key: convert_to_device(features_dct[key]) for key in features_dct}
            test_labels = test_labels.to(DEVICE)
            prediction = NN(features_dct)
            test_acc += (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
            predictions.append(prediction)
            i += 1

    print(f"Test accuracy of {test_acc / i}")
    return predictions, (test_acc / i)


def run(processor_type, NN, output_suffix):
    model_parameters = filter(lambda p: p.requires_grad, NN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    train_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                               shuffle=True, pin_memory=True)
    write_file_name = f"./results/{train_data.get_name()}_{NN.__class__.__name__}_{output_suffix}"

    test_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=4,
                              shuffle=True, pin_memory=True)


    optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    average_time, loss, final_loss, train_acc = train(NN, 100, train_dataset, optimizer, criterion, processor_type)

    _, test_acc = test(NN, test_dataset, processor_type)

    s = f"""Dataset: {dataset},\nModel: {NN.__class__.__name__}\n\nparams={params}\n\n
    FINAL RESULTS\nTEST ACCURACY: {test_acc:.4f}\nTRAIN ACCURACY: {train_acc:.4f}\n\n
    Average Time Taken: {timedelta(seconds=average_time)}\nAverage Loss: {loss:.4f}\n\n\n"""

    print(s)

    with open(write_file_name + '.txt', 'w') as f:
        f.write(s)


if __name__ == "__main__":
    # NN_list = [gnn, gat, Ebli_nn, Bunch_nn]
    NN_list = [sat_nn]
    for output_suffix in range(5):
        for processor_type, NN in NN_list:
            NN = NN(5, 10, 15, output_size)
            run(processor_type, NN.to(DEVICE), output_suffix)
