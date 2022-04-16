from Superpixel.SuperpixelDataset import SuperpixelSCDataset
from Superpixel.EdgeFlow import PixelBasedEdgeFlow, RandomBasedEdgeFlow
from Superpixel.ImageProcessor import ImageProcessor, OrientatedImageProcessor
from torch.utils.data import DataLoader
from constants import DEVICE
from models import superpixel_GCN, superpixel_GAT, superpixel_ESNN, superpixel_BSNN, superpixel_SAT, superpixel_SAN
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
# edgeFlow = RandomBasedEdgeFlow
imageprocessor = ImageProcessor
# imageprocessor = OrientatedImageProcessor
full_dataset = 60000
train_set = 55000
val_set = 5000
test_set = 10000

output_size = 10

# model_list = [superpixel_GCN, superpixel_GAT, superpixel_ESNN, superpixel_BSNN, superpixel_SAT, superpixel_SAN]
model = superpixel_GCN


def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]


def top_n_error_rate(prediction, actual, n):
    pred = prediction.topk(n, 1, largest=True, sorted=True)
    predictions = pred.indices
    predictions = predictions - actual.unsqueeze(1)
    return (predictions.prod(dim=1) == 0).float().mean().item()


def train(NN, epoch_size, train_data, optimizer, criterion, processor_type):
    train_running_loss = 0
    t = 0
    best_val_acc = 0
    train_dataset, val_dataset = train_data.get_val_train_split(full_dataset, train_set)
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
        for simplicialComplex, train_labels in train_dataset:
            simplicialComplex.to_device()
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            train_labels = train_labels.to(DEVICE)
            optimizer.zero_grad()
            prediction = NN(simplicialComplex)
            loss = criterion(prediction, train_labels)
            loss.backward()
            optimizer.step()
            epoch_train_running_loss += loss.detach().item()
            train_acc = (torch.argmax(prediction, 1).flatten() == train_labels).type(torch.float).mean().item()
            i += 1
            training_acc += (train_acc - training_acc) / i
        t2 = time.perf_counter()
        NN.eval()
        for simplicialComplex, val_labels in val_dataset:
            simplicialComplex.to_device()
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            val_labels = val_labels.to(DEVICE)
            prediction = NN(simplicialComplex)
            val_acc = (torch.argmax(prediction, 1).flatten() == val_labels).type(torch.float).mean().item()
            j += 1
            validation_acc += (val_acc - validation_acc) / j
        t = (t * epoch + (t2 - t1)) / (epoch + 1)
        epoch_train_running_loss /= i
        train_running_loss = (train_running_loss * epoch + epoch_train_running_loss) / (epoch + 1)
        if validation_acc > best_val_acc:
            torch.save(NN.state_dict(), f'./data/{NN.__class__.__name__}_nn.pkl')
            best_val_acc = validation_acc
            associated_training_acc = training_acc
        print(
            f"Epoch {epoch}"
            f"| Train accuracy {training_acc:.4f} | Validation accuracy {validation_acc:.4f}")
    return t, associated_training_acc, best_val_acc


def test(NN, dataloader, processor_type):
    NN.eval()

    test_acc = 0
    i = 0
    top_3_error = 0
    top_5_error = 0
    predictions = []
    with torch.no_grad():
        for simplicialComplex, test_labels in dataloader:
            simplicialComplex.to_device()
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            test_labels = test_labels.to(DEVICE)
            prediction = NN(simplicialComplex)
            top_3_error += top_n_error_rate(prediction, test_labels, 3)
            top_5_error += top_n_error_rate(prediction, test_labels, 5)
            test_acc += (torch.argmax(prediction, 1).flatten() == test_labels).type(torch.float).mean().item()
            predictions.append(prediction)
            i += 1

    print(
        f"Test accuracy of {test_acc / i}, top 3 error rate of {1 - (top_3_error / i)}, top 5 error rate of {1 - (top_5_error / i)}")
    return (test_acc / i), top_3_error / i, top_5_error / i, predictions


def run(processor_type, NN, output_suffix):
    model_parameters = filter(lambda p: p.requires_grad, NN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    train_data = SuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type, imageprocessor,
                                     full_dataset, train=True)

    write_file_name = f"./results/{train_data.get_name()}_{NN.__class__.__name__}_{output_suffix}"

    test_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, imageprocessor,
                                    test_set, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                              shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    average_time, train_acc, val_acc = train(NN, 100, train_data, optimizer, criterion, processor_type)

    NN.load_state_dict(torch.load(f'./data/{NN.__class__.__name__}_nn.pkl'))
    test_acc, top_3_error, top_5_error, _ = test(NN, test_dataset, processor_type)
    print(test_acc, top_3_error, top_5_error)

    s = f"""Dataset: {dataset},\nModel: {NN.__class__.__name__}\n\nparams={params}\n\n
    FINAL RESULTS\nTEST ACCURACY: {test_acc:.4f}\nTRAIN ACCURACY: {train_acc:.4f}\n\n
    Average Time Taken: {timedelta(seconds=average_time)}\n\n
    Top 3 error rate of {1 - top_3_error:.4f}, top 5 error rate of {1 - top_5_error:.4f}"""

    print(s)

    with open(write_file_name + '.txt', 'w') as f:
        f.write(s)


if __name__ == "__main__":
    for output_suffix in range(1):
        processor_type, NN = model
        NN = NN(5, output_size)
        run(processor_type, NN.to(DEVICE), output_suffix)
