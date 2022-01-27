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
from datetime import timedelta
import time
import torchvision.transforms as transforms
from joblib import Parallel, delayed

batch_size = 8
superpixel_size = 75
dataset = datasets.MNIST
# dataset = datasets.CIFAR10
edgeFlow = PixelBasedEdgeFlow

output_size = 10
epsilons = [0, .05, .1, .15, .2, .25, .3]

def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]

def make_smaller_dataset_10_classes(data):
    l = len(data)
    data = [*sorted(data, key=lambda i: i[1])]
    data_out = []
    for i in range(10):
        data_out += data[i * l // 10: (i * 5 + 1) * l // 50]
    return data_out


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


def change_offset_of_edge_and_tri(X, batch, node_offset, shape):
    batch_unique = batch.unique(sorted=True)
    batch_unique_count = torch.stack([(batch == batch_u).sum() for batch_u in batch_unique]).tolist()
    new_offset = []
    acc = 0
    for n_offset, other_offset in zip(node_offset, batch_unique_count):
        new_offset.append(torch.ones((other_offset, shape)) * acc)
        acc += n_offset
    X_offset = torch.cat(new_offset, dim = 0).long()
    X += X_offset
    return X


def add_edge_and_tri_offset(feature_tuple):
    feature_dct, label = feature_tuple

    batch_0 = feature_dct['batch_index'][0]
    batch_unique_0 = batch_0 .unique(sorted=True)
    batch_unique_0_count = torch.stack([(batch_0  == batch_u).sum() for batch_u in batch_unique_0 ]).tolist()
    batch_unique_0_offset = batch_unique_0_count

    if len(feature_dct['features']) > 1:
        X1 = feature_dct['features'][1]
        batch1 = feature_dct['batch_index'][1]
        feature_dct['features'][1] = change_offset_of_edge_and_tri(X1, batch1, batch_unique_0_offset, 2)

        X2 = feature_dct['features'][2]
        batch2 = feature_dct['batch_index'][2]
        feature_dct['features'][2] = change_offset_of_edge_and_tri(X2, batch2, batch_unique_0_offset, 3)

    return feature_dct, label

def add_edge_and_tri_features(feature_dct):
    if len(feature_dct['features']) > 1:
        X0 = feature_dct['features'][0]
        X1 = feature_dct['features'][1]
        X2 = feature_dct['features'][2]

        X1_i, X1_j = X0[X1[:,0]], X0[X1[:,1]]
        X1 = torch.cat([X1_i, X1_j], dim = 1)
        feature_dct['features'][1] = X1

        X2_i, X2_j, X2_k = X0[X2[:,0]], X0[X2[:,1]], X0[X2[:,2]]
        X2 = torch.cat([X2_i, X2_j, X2_k], dim = 1)
        feature_dct['features'][2] = X2
    return feature_dct


def create_node_mask(feature_dct, masks):
    batch_0 = feature_dct['batch_index'][0]
    batch_unique_0 = batch_0.unique(sorted=True)
    batch_unique_0_count = torch.stack([(batch_0 == batch_u).sum() for batch_u in batch_unique_0]).tolist()

    batch_mask = []
    acc = 0
    for offset, mask in zip(batch_unique_0_count, masks):
        batch_mask.append(torch.ones((offset, 3)) * mask)
        acc += offset
    batch_mask = torch.cat(batch_mask, dim = 0)

    return batch_mask


def fgsm_attack(feature_dct, epsilon, data_grad, batch_mask, targeted = False):
    pixel_index = torch.tensor([0,1,2])
    data_grad = torch.index_select(data_grad.cpu(), 1, pixel_index)
    sign_data_grad = data_grad.sign() * batch_mask * (1 - int(targeted) * 2)
    X0 = feature_dct['features'][0].cpu()
    X0_coordinates = torch.index_select(X0, 1, torch.tensor([3, 4]))
    X0_pixels = torch.index_select(X0, 1, pixel_index) + epsilon*sign_data_grad
    X0_pixels = torch.clamp(X0_pixels, 0, 1)
    X0 = torch.cat([X0_pixels, X0_coordinates], dim=1)
    feature_dct['features'][0] = X0
    return feature_dct


def set_grad(feature_dct):
    for i in range(len(feature_dct['features'])):
        feature_dct['features'][i].requires_grad = True
    return feature_dct


def get_grad(feature_dct, original_edge_indexes, original_tri_indexes):
    if len(feature_dct['features']) == 1:
        return feature_dct['features'][0].grad.data
    else:
        X0_grad = feature_dct['features'][0].grad.data.cpu()
        X1_grad = feature_dct['features'][1].grad.data.cpu()
        X2_grad = feature_dct['features'][2].grad.data.cpu()

        X1_slices = [[i for i in range(5)], [i for i in range(5, 10)]]
        X2_slices = [[i for i in range(5)], [i for i in range(5, 10)], [i for i in range(10, 15)]]

        for i in range(2):
            index = original_edge_indexes[:,i]
            slice = torch.index_select(X1_grad, 1, torch.tensor(X1_slices[i]))
            X0_grad = X0_grad.index_add_(0, index, slice)

        for i in range(3):
            index = original_tri_indexes[:,i]
            slice = torch.index_select(X2_grad, 1, torch.tensor(X2_slices[i]))
            X0_grad = X0_grad.index_add_(0, index, slice)

        return X0_grad

def test(NN, dataloader, batch_size, epsilon = 0.002, targeted = True):
    NN.eval()
    no_epoch = 50
    full_batched_feature_dct, full_test_labels = [*map(list, zip(*dataloader))]
    full_target_labels = [(labels + torch.randint(1, 9, (labels.shape[0], )))%10 for labels in full_test_labels]

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
            initial_pred = prediction.argmax(dim = 1)
            initial_acc += test_labels.eq(initial_pred).sum().item()/batch_size

            if targeted:
                filter_indexes = [int(initial_pred[i] != target_labels[i]) for i in range(batch_size)]
                loss = torch.nn.functional.nll_loss(prediction, target_labels)
            else:
                filter_indexes = [int(initial_pred[i] == test_labels[i]) for i in range(batch_size)]
                loss = torch.nn.functional.nll_loss(prediction, test_labels)

            NN.zero_grad()
            loss.backward()

            if len(feature_dct['features']) == 1:
                data_grad = feature_dct['features'][0].grad.data
            else:
                X0_grad = feature_dct['features'][0].grad.data.cpu()
                X1_grad = feature_dct['features'][1].grad.data.cpu()
                X2_grad = feature_dct['features'][2].grad.data.cpu()

                X1_slices = [[i for i in range(5)], [i for i in range(5, 10)]]
                X2_slices = [[i for i in range(5)], [i for i in range(5, 10)], [i for i in range(10, 15)]]

                for j in range(2):
                    index = batched_feature_dct['features'][1][:, j]
                    slice = torch.index_select(X1_grad, 1, torch.tensor(X1_slices[j]))
                    X0_grad = X0_grad.index_add_(0, index, slice)

                for j in range(3):
                    index = batched_feature_dct['features'][2][:, j]
                    slice = torch.index_select(X2_grad, 1, torch.tensor(X2_slices[j]))
                    X0_grad = X0_grad.index_add_(0, index, slice)

                data_grad = X0_grad

            feature_dct = copy.deepcopy(batched_feature_dct)
            node_mask = create_node_mask(feature_dct, filter_indexes)
            feature_dct = fgsm_attack(feature_dct, epsilon, data_grad, node_mask, targeted)
            feature_dct = add_edge_and_tri_features(feature_dct)
            feature_dct = {key: convert_to_device(feature_dct[key]) for key in feature_dct}
            prediction = NN(feature_dct)

            final_pred = prediction.argmax(dim = 1)
            final_acc += final_pred.eq(test_labels).sum().item()/batch_size
            adv_acc += final_pred.eq(target_labels).sum().item()/batch_size

            batched_feature_dct['features'][0] = feature_dct['features'][0]

        if epoch == 0:
            start_initial_acc = initial_acc / (i + 1)

        final_acc /= (i + 1)
        adv_acc /= (i + 1)

        print(start_initial_acc, final_acc, adv_acc)

    print(f"Initial accuracy of {start_initial_acc}, final accuracy of {final_acc}, adv accuracy of {adv_acc}")
    return initial_acc, final_acc


def run(processor_type, NN):
    model_parameters = filter(lambda p: p.requires_grad, NN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    train_data = SuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type, train=True)
    train_dataset = DataLoader(train_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                               shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # average_time, loss, final_loss, train_acc = train(NN, 100, train_dataset, optimizer, criterion, processor_type)
    # print(train_acc)
    # torch.save(NN.state_dict(), f'../data/{NN.__class__.__name__}_nn.pkl')
    NN.load_state_dict(torch.load(f'../data/{NN.__class__.__name__}_nn.pkl'))
    test_data = AdversarialSuperpixelSCDataset('../data', dataset, superpixel_size, edgeFlow, processor_type, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8, shuffle=True, pin_memory=True)
    test_dataset = [data for data in test_dataset]
    test_dataset = [*map(lambda x : (processor_type.clean_feature_dct(x[0]), x[1]), test_dataset)]
    test_dataset = [*map(lambda x : (processor_type.repair(x[0]), x[1]), test_dataset)]
    test_dataset = [*map(add_edge_and_tri_offset, test_dataset)]
    final_acc = test(NN, test_dataset, batch_size)


if __name__ == "__main__":
    # NN_list = [superpixel_gnn, superpixel_gat, superpixel_Ebli_nn, superpixel_Bunch_nn, superpixel_sat_nn]
    NN_list = [superpixel_sat_nn]
    for processor_type, NN in NN_list:
        NN = NN(5, 10, 15, output_size)
        run(processor_type, NN.to(DEVICE))
