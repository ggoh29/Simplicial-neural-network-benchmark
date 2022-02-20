from Superpixel.SuperpixelDataset.SuperpixelLoader import SuperpixelSCDataset
from Superpixel.SuperpixelDataset.EdgeFlow import PixelBasedEdgeFlow
from torch.utils.data import DataLoader
from constants import DEVICE
from models import superpixel_sat_nn, superpixel_gat, superpixel_gnn, superpixel_Bunch_nn, superpixel_Ebli_nn
import torch
from torchvision import datasets
import numpy as np
from datetime import timedelta
import time

batch_size = 32
superpixel_size = 75
dataset = datasets.MNIST
# dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow
# edgeFlow = RandomBasedEdgeFlow;

output_size = 10

def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]

def train(NN, epoch_size, train_data, optimizer, criterion, processor_type):
    train_running_loss = 0
    t = 0
    best_val_acc = 0
    train_dataset, val_dataset = train_data.get_val_train_split()
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                               shuffle=True, pin_memory=True)
    val_dataset = DataLoader(val_dataset, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                             shuffle=True, pin_memory=True)

    for epoch in range(epoch_size):
        t1 = time.perf_counter()
        epoch_train_running_loss = 0
        train_acc, training_acc = 0, 0
        val_acc, validation_acc = 0, 0
        i, j = 0, 0
        NN.train()
        for features_dct, train_labels in train_dataset:
            features_dct = processor_type.clean_feature_dct(features_dct)
            features_dct = processor_type.repair(features_dct)
            features_dct = {key: convert_to_device(features_dct[key]) for key in features_dct}
            train_labels = train_labels.to(DEVICE)
            optimizer.zero_grad()
            prediction = NN(features_dct)
            loss = criterion(prediction, train_labels)
            loss.backward()
            optimizer.step()
            epoch_train_running_loss += loss.detach().item()
            train_acc = (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
            i += 1
            training_acc += (train_acc - training_acc)/i
        t2 = time.perf_counter()
        NN.eval()
        for features_dct, val_labels in val_dataset:
            features_dct = processor_type.clean_feature_dct(features_dct)
            features_dct = processor_type.repair(features_dct)
            features_dct = {key: convert_to_device(features_dct[key]) for key in features_dct}
            val_labels = val_labels.to(DEVICE)
            prediction = NN(features_dct)
            val_acc = (torch.argmax(prediction, 1).flatten() == val_labels).type(torch.float).mean().item()
            j += 1
            validation_acc += (val_acc - validation_acc)/j
        t = (t * epoch + (t2 - t1)) / (epoch + 1)
        epoch_train_running_loss /= i
        train_running_loss = (train_running_loss * epoch + epoch_train_running_loss) / (epoch + 1)
        if validation_acc > best_val_acc:
            torch.save(NN.state_dict(), f'./data/{NN.__class__.__name__}_nn.pkl')
        print(
            f"Epoch {epoch} | Running loss {train_running_loss} "
            f"| Train accuracy {training_acc} | Validation accuracy {validation_acc}")
    return t, train_running_loss, training_acc, validation_acc


def test(NN, dataloader, processor_type):
    NN.eval()

    test_acc = 0
    i = 0
    predictions = []
    with torch.no_grad():
        for features_dct, test_labels in dataloader:
            features_dct = processor_type.clean_feature_dct(features_dct)
            features_dct = processor_type.repair(features_dct)
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

    train_data = SuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type, train=True)

    write_file_name = f"./results/{train_data.get_name()}_{NN.__class__.__name__}_{output_suffix}"

    test_data = SuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                              shuffle=True, pin_memory=True)


    optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    average_time, loss, train_acc, val_acc = train(NN, 100, train_data, optimizer, criterion, processor_type)

    NN.load_state_dict(torch.load(f'./data/{NN.__class__.__name__}_nn.pkl'))
    _, test_acc = test(NN, test_dataset, processor_type)

    s = f"""Dataset: {dataset},\nModel: {NN.__class__.__name__}\n\nparams={params}\n\n
    FINAL RESULTS\nTEST ACCURACY: {test_acc:.4f}\nTRAIN ACCURACY: {train_acc:.4f}\n\n
    Average Time Taken: {timedelta(seconds=average_time)}\nAverage Loss: {loss:.4f}\n\n\n"""

    print(s)

    with open(write_file_name + '.txt', 'w') as f:
        f.write(s)


if __name__ == "__main__":
    # NN_list = [superpixel_gnn, superpixel_gat, superpixel_Ebli_nn, superpixel_Bunch_nn]
    NN_list = [superpixel_gnn]
    for output_suffix in range(1):
        for processor_type, NN in NN_list:
            NN = NN(5, output_size)
            run(processor_type, NN.to(DEVICE), output_suffix)
