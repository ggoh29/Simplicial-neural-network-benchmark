import copy

from AdversarialSuperpixelDataset.SuperpixelLoader import SuperpixelSCDataset
from AdversarialSuperpixelDataset.AdversarialSuperpixelLoader import AdversarialSuperpixelSCDataset
from AdversarialSuperpixelDataset.EdgeFlow import PixelBasedEdgeFlow
from torch.utils.data import DataLoader
from constants import DEVICE
from models import superpixel_sat_nn, superpixel_gat, superpixel_gnn, superpixel_Bunch_nn, superpixel_Ebli_nn
import torch
from torchvision import datasets
import numpy as np
import time
from AdversarialSuperpixelDataset.fgsm import aggregate_grad, set_grad, add_edge_and_tri_features, fgsm_attack, \
    add_edge_and_tri_offset
import os

batch_size = 8
superpixel_size = 75
dataset = datasets.MNIST
edgeFlow = PixelBasedEdgeFlow

output_size = 10


def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]


def make_smaller_dataset_10_classes(data):
    l = len(data)
    data = [*sorted(data, key=lambda i: i[1])]
    data_out = []
    for i in range(10):
        data_out += data[i * l // 10: (i * 5 + 1) * l // 50]
    return data_out

def load_trained_NN(NN, dataset, processor_type):
    if not os.path.isfile(f'../data/{NN.__class__.__name__}_nn.pkl'):
        train_data = SuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
        train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                                   shuffle=True, pin_memory=True)

        optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        average_time, loss, final_loss, train_acc = train(NN, 100, train_dataset, optimizer, criterion, processor_type)
        print(train_acc)
        torch.save(NN.state_dict(), f'../data/{NN.__class__.__name__}_nn.pkl')
    NN.load_state_dict(torch.load(f'../data/{NN.__class__.__name__}_nn.pkl'))
    return NN


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
            features_dct = processor_type.repair(features_dct)
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


def gen_adversarial_dataset(NN, dataloader, full_target_labels, batch_size, epsilon=0.002, targeted=True):
    NN.eval()
    no_epoch = 50
    full_batched_feature_dct, full_test_labels = [*map(list, zip(*dataloader))]

    start_initial_acc = 0

    for epoch in range(no_epoch):
        initial_acc = 0
        final_acc = 0
        adv_acc = 0
        for i in range(len(dataloader)):
            batched_feature_dct, test_labels = full_batched_feature_dct[i], full_test_labels[i]
            target_labels = full_target_labels[i]

            target_labels, test_labels = target_labels.to(DEVICE), test_labels.to(DEVICE)
            feature_dct = copy.deepcopy(batched_feature_dct)
            feature_dct = add_edge_and_tri_features(feature_dct)
            feature_dct = {key: convert_to_device(feature_dct[key]) for key in feature_dct}

            feature_dct = set_grad(feature_dct)
            prediction = NN(feature_dct)
            initial_pred = prediction.argmax(dim=1)
            initial_acc += test_labels.eq(initial_pred).sum().item() / batch_size
            if targeted:
                loss = torch.nn.functional.nll_loss(prediction, target_labels)
            else:
                loss = torch.nn.functional.nll_loss(prediction, test_labels)

            NN.zero_grad()
            loss.backward()

            if len(feature_dct['features']) == 1:
                data_grad = feature_dct['features'][0].grad.data
            else:
                X0_grad = feature_dct['features'][0].grad.data.cpu()
                X1_grad = feature_dct['features'][1].grad.data.cpu()
                X2_grad = feature_dct['features'][2].grad.data.cpu()

                X1_grad_copy = torch.zeros(X0_grad.shape)
                X2_grad_copy = torch.zeros(X0_grad.shape)

                X0_grad += aggregate_grad(batched_feature_dct['features'][1], X1_grad, X1_grad_copy, 2)
                X0_grad += aggregate_grad(batched_feature_dct['features'][2], X2_grad, X2_grad_copy, 3)

                data_grad = X0_grad

            feature_dct = copy.deepcopy(batched_feature_dct)
            feature_dct = fgsm_attack(feature_dct, epsilon, data_grad, targeted)
            feature_dct = add_edge_and_tri_features(feature_dct)
            feature_dct = {key: convert_to_device(feature_dct[key]) for key in feature_dct}
            prediction = NN(feature_dct)

            final_pred = prediction.argmax(dim=1)
            final_acc += final_pred.eq(test_labels).sum().item() / batch_size
            adv_acc += final_pred.eq(target_labels).sum().item() / batch_size

            batched_feature_dct['features'][0] = feature_dct['features'][0]

        if epoch == 0:
            start_initial_acc = initial_acc / (i + 1)

        final_acc /= (i + 1)
        adv_acc /= (i + 1)


    print(f"Initial accuracy of {start_initial_acc}, final accuracy of {final_acc}, target accuracy of {adv_acc} of {NN.__class__.__name__}")
    return initial_acc, final_acc, full_batched_feature_dct


def run_test_set(NN, dataloader, full_target_labels, batch_size):
    NN.eval()
    full_batched_feature_dct, full_test_labels = [*map(list, zip(*dataloader))]

    final_acc = 0
    adv_acc = 0

    for i in range(len(dataloader)):
        batched_feature_dct, test_labels = full_batched_feature_dct[i], full_test_labels[i]
        target_labels = full_target_labels[i]

        target_labels, test_labels = target_labels.to(DEVICE), test_labels.to(DEVICE)
        feature_dct = copy.deepcopy(batched_feature_dct)
        feature_dct = add_edge_and_tri_features(feature_dct)
        feature_dct = {key: convert_to_device(feature_dct[key]) for key in feature_dct}

        feature_dct = set_grad(feature_dct)
        prediction = NN(feature_dct)
        initial_pred = prediction.argmax(dim=1)
        final_acc += initial_pred.eq(test_labels).sum().item() / batch_size
        adv_acc += initial_pred.eq(target_labels).sum().item() / batch_size

    final_acc /= (i + 1)
    adv_acc /= (i + 1)

    print(f"Final accuracy of {final_acc}, target accuracy of {adv_acc} of {NN.__class__.__name__}")


def run_direct_attack(processor_type, NN):
    model_parameters = filter(lambda p: p.requires_grad, NN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    NN = load_trained_NN(NN, dataset, processor_type)

    test_data = AdversarialSuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type,
                                               train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                              shuffle=False, pin_memory=True)
    test_dataset = [data for data in test_dataset]
    test_dataset = [*map(lambda x: (processor_type.clean_feature_dct(x[0]), x[1]), test_dataset)]
    test_dataset = [*map(lambda x: (processor_type.repair(x[0]), x[1]), test_dataset)]
    _, full_test_labels = [*map(list, zip(*test_dataset))]
    target_labels = [(labels + torch.randint(1, 9, (labels.shape[0],))) % 10 for labels in full_test_labels]
    test_dataset = [*map(add_edge_and_tri_offset, test_dataset)]
    gen_adversarial_dataset(NN, test_dataset,target_labels,  batch_size)


def run_transferability_attack(base_nn, base_processor_type, target_nn, target_processor_type):
    """Test the attack transferability of perturbed images in which the adversarial images are generated for base_nn but
    used against target_nn"""
    base_nn = load_trained_NN(base_nn, dataset, base_processor_type)
    base_test_data = AdversarialSuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, base_processor_type, train=False)
    base_test_dataset = DataLoader(base_test_data, batch_size=batch_size, collate_fn=base_processor_type.batch, num_workers=8, shuffle=False, pin_memory=True)
    base_test_dataset = [data for data in base_test_dataset]
    base_test_dataset = [*map(lambda x: (base_processor_type.clean_feature_dct(x[0]), x[1]), base_test_dataset)]
    base_test_dataset = [*map(lambda x: (base_processor_type.repair(x[0]), x[1]), base_test_dataset)]
    _, full_test_labels = [*map(list, zip(*base_test_dataset))]
    target_labels = [(labels + torch.randint(1, 9, (labels.shape[0],))) % 10 for labels in full_test_labels]
    base_test_dataset = [*map(add_edge_and_tri_offset, base_test_dataset)]
    _, _, full_batched_feature_dct = gen_adversarial_dataset(base_nn, base_test_dataset, target_labels, batch_size)

    target_nn = load_trained_NN(target_nn, dataset, target_processor_type)
    target_test_data = AdversarialSuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, target_processor_type, train=False)
    target_test_dataset = DataLoader(target_test_data, batch_size=batch_size, collate_fn=target_processor_type.batch, num_workers=8, shuffle=False, pin_memory=True)
    target_test_dataset = [data for data in target_test_dataset]
    target_test_dataset = [*map(lambda x: (target_processor_type.clean_feature_dct(x[0]), x[1]), target_test_dataset)]
    target_test_dataset = [*map(lambda x: (target_processor_type.repair(x[0]), x[1]), target_test_dataset)]
    target_test_dataset = [*map(add_edge_and_tri_offset, target_test_dataset)]
    print(f"Accuracy of {target_nn.__class__.__name__} before transfer attack from {base_nn.__class__.__name__}")
    run_test_set(target_nn, target_test_dataset, target_labels, batch_size)

    target_feature_dct, full_test_labels = [*map(list, zip(*target_test_dataset))]
    for i in range(len(target_feature_dct)):
        target_feature_dct[i]['features'][0] = full_batched_feature_dct[i]['features'][0]
    target_test_dataset = [*zip(target_feature_dct, full_test_labels)]
    print(f"Accuracy of {target_nn.__class__.__name__} after transfer attack from {base_nn.__class__.__name__}")
    run_test_set(target_nn, target_test_dataset, target_labels, batch_size)


if __name__ == "__main__":
    # NN_list = [superpixel_gnn, superpixel_gat, superpixel_Ebli_nn, superpixel_Bunch_nn, superpixel_sat_nn]
    NN_list = [superpixel_Ebli_nn]
    for _ in range(5):
        for processor_type, NN in NN_list:
            NN = NN(5, 10, 15, output_size)
            run_direct_attack(processor_type, NN.to(DEVICE))
    # for _ in range(5):
    #     base_processor_type, base_nn = superpixel_Ebli_nn
    #     target_processor_type, target_nn = superpixel_gnn
    #     base_nn = base_nn(5, 10, 15, output_size).to(DEVICE)
    #     target_nn = target_nn(5, output_size).to(DEVICE)
    #     run_transferability_attack(base_nn, base_processor_type, target_nn, target_processor_type)
    #     print()


