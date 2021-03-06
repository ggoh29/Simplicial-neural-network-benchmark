import copy
from Superpixel.SuperpixelDataset import SuperpixelSCDataset
from Superpixel.EdgeFlow import PixelBasedEdgeFlow
from Superpixel.ImageProcessor import ImageProcessor, AdversarialImageProcessor
from torch.utils.data import DataLoader
from constants import DEVICE
from models import superpixel_GCN, superpixel_GAT, superpixel_SCN, superpixel_SCConv, superpixel_SAT, superpixel_SAN
import torch
from torchvision import datasets
import numpy as np
import time
from Superpixel.adversarialSuperpixel.fgsm import aggregate_grad, set_grad, add_edge_and_tri_features, fgsm_attack, \
    add_edge_and_tri_offset
import os
from tqdm import tqdm

batch_size = 32
superpixel_size = 75
dataset = datasets.MNIST
edgeFlow = PixelBasedEdgeFlow
output_size = 10

full_dataset = 12000
train_set = 10000
val_set = 2000
test_set = 2000

targeted = True
direct_attack = True
# alternative is transfer attack

model_list = [superpixel_GCN, superpixel_GAT, superpixel_SCN, superpixel_SCConv, superpixel_SAT, superpixel_SAN]

model = superpixel_GCN
# model = superpixel_GAT
# model = superpixel_SCN
# model = superpixel_SCConv
# model = superpixel_SAT
# model = superpixel_SAN


def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]


def load_trained_NN(NN, dataset, processor_type):
    if not os.path.isfile(f'./data/{NN.__class__.__name__}_nn.pkl'):
        train_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor,
                                         12000, train=True)

        optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        _ = train(NN, 200, train_data, optimizer, criterion, processor_type)

    NN.load_state_dict(torch.load(f'./data/{NN.__class__.__name__}_nn.pkl'))
    return NN


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
            simplicialComplex = processor_type.clean_features(simplicialComplex)
            simplicialComplex = processor_type.repair(simplicialComplex)
            simplicialComplex.to_device()
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


def gen_adversarial_dataset(NN, dataloader, target_labels_list, batch_size, epsilon=0.001, targeted=False,
                            no_epoch=250):
    NN.eval()
    simplicialComplex_list, test_labels_list = [*map(list, zip(*dataloader))]

    start_initial_acc = 0

    gradient = []
    acc = []
    loss_f = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(no_epoch)):
        m = []
        initial_acc = 0
        final_acc = 0
        adv_acc = 0
        for i in range(len(dataloader)):
            simplicialComplex_original, test_labels = simplicialComplex_list[i], test_labels_list[i]
            target_labels = target_labels_list[i]

            target_labels, test_labels = target_labels.to(DEVICE), test_labels.to(DEVICE)
            simplicialComplex = copy.deepcopy(simplicialComplex_original)
            simplicialComplex = add_edge_and_tri_features(simplicialComplex)
            simplicialComplex.to_device()

            set_grad(simplicialComplex)
            prediction = NN(simplicialComplex)
            initial_pred = prediction.argmax(dim=1)
            initial_acc += test_labels.eq(initial_pred).sum().item() / batch_size
            if targeted:
                loss = loss_f(prediction, target_labels)
            else:
                loss = loss_f(prediction, test_labels)

            NN.zero_grad()
            loss.backward()

            if simplicialComplex.X1 is None:
                data_grad = simplicialComplex.X0.grad.data
            else:
                X0_grad = simplicialComplex.X0.grad.data.cpu()
                X1_grad = simplicialComplex.X1.grad.data.cpu()
                X2_grad = simplicialComplex.X2.grad.data.cpu()

                X1_grad_copy = torch.zeros(X0_grad.shape)
                X2_grad_copy = torch.zeros(X0_grad.shape)

                X0_grad += aggregate_grad(simplicialComplex_original.X1, X1_grad, X1_grad_copy, 2)
                X0_grad += aggregate_grad(simplicialComplex_original.X2, X2_grad, X2_grad_copy, 3)

                data_grad = X0_grad

            m.append(data_grad)

            simplicialComplex = copy.deepcopy(simplicialComplex_original)
            simplicialComplex = fgsm_attack(simplicialComplex, epsilon, data_grad, targeted)
            simplicialComplex = add_edge_and_tri_features(simplicialComplex)
            simplicialComplex.to_device()
            prediction = NN(simplicialComplex)

            final_pred = prediction.argmax(dim=1)
            final_acc += final_pred.eq(test_labels).sum().item() / batch_size
            adv_acc += final_pred.eq(target_labels).sum().item() / batch_size

            simplicialComplex_original.X0 = simplicialComplex.X0.cpu()

        if epoch == 0:
            start_initial_acc = initial_acc / (i + 1)

        final_acc /= (i + 1)

        try:
            ac = final_acc / start_initial_acc
        except ZeroDivisionError:
            ac = 0

        acc.append(ac)
        adv_acc /= (i + 1)

        if (epoch + 1) % 50 == 0:
            print(
                f"\nInitial accuracy of {start_initial_acc}, final accuracy of {final_acc}, target accuracy of {adv_acc} of {NN.__class__.__name__}")

        # m = torch.cat(m, dim=0)
        # m = m.cpu()
        # m = torch.sum(m.pow(2))
        # gradient.append(m.item())

    # print(acc)
    # print(gradient)
    print(
        f"Initial accuracy of {start_initial_acc}, final accuracy of {final_acc}, target accuracy of {adv_acc} of {NN.__class__.__name__}")
    return initial_acc, final_acc, simplicialComplex_list


def run_test_set(NN, dataloader, full_target_labels, batch_size):
    NN.eval()
    simplicialComplex_list, full_test_labels = [*map(list, zip(*dataloader))]

    final_acc = 0
    adv_acc = 0

    for i in range(len(dataloader)):
        simplicialComplex_original, test_labels = simplicialComplex_list[i], full_test_labels[i]
        target_labels = full_target_labels[i]

        target_labels, test_labels = target_labels.to(DEVICE), test_labels.to(DEVICE)
        simplicialComplex = copy.deepcopy(simplicialComplex_original)
        simplicialComplex = add_edge_and_tri_features(simplicialComplex)

        simplicialComplex = set_grad(simplicialComplex)
        simplicialComplex.to_device()
        prediction = NN(simplicialComplex)
        initial_pred = prediction.argmax(dim=1)
        final_acc += initial_pred.eq(test_labels).sum().item() / batch_size
        adv_acc += initial_pred.eq(target_labels).sum().item() / batch_size

    final_acc /= (i + 1)
    adv_acc /= (i + 1)

    print(f"Final accuracy of {final_acc}, target accuracy of {adv_acc} of {NN.__class__.__name__}")


def run_direct_attack(processor_type, NN, targeted=False):
    model_parameters = filter(lambda p: p.requires_grad, NN.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    NN = load_trained_NN(NN, dataset, processor_type)

    test_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type,
                                    AdversarialImageProcessor, 2000, train=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, collate_fn=processor_type.batch, num_workers=8,
                              shuffle=False, pin_memory=True)
    test_dataset = [data for data in test_dataset]
    test_dataset = [*map(lambda x: (processor_type.clean_features(x[0]), x[1]), test_dataset)]
    test_dataset = [*map(lambda x: (processor_type.repair(x[0]), x[1]), test_dataset)]
    _, full_test_labels = [*map(list, zip(*test_dataset))]
    target_labels = [(labels + torch.randint(1, 9, (labels.shape[0],))) % 10 for labels in full_test_labels]
    test_dataset = [*map(add_edge_and_tri_offset, test_dataset)]
    gen_adversarial_dataset(NN, test_dataset, target_labels, batch_size, targeted=targeted)


def gen_transferability_attack(base_nn, base_processor_type, epsilon=0.001, targeted=True, full_batched_dataset=None,
                               target_labels=None, no_epoch=50):
    """Test the attack transferability of perturbed images in which the adversarial images are generated for base_nn but
    used against target_nn"""
    base_nn = load_trained_NN(base_nn, dataset, base_processor_type)
    if full_batched_dataset is not None:
        full_batched_dataset = [(data, label) for data, label in zip(full_batched_dataset, target_labels)]
        _, _, full_batched_dataset = gen_adversarial_dataset(base_nn, full_batched_dataset, target_labels, batch_size,
                                                             epsilon, targeted, no_epoch=no_epoch)
    else:
        base_test_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, base_processor_type,
                                             AdversarialImageProcessor, 2000, train=False)
        base_test_dataset = DataLoader(base_test_data, batch_size=batch_size, collate_fn=base_processor_type.batch,
                                       num_workers=8, shuffle=False, pin_memory=True)
        base_test_dataset = [data for data in base_test_dataset]
        base_test_dataset = [*map(lambda x: (base_processor_type.clean_features(x[0]), x[1]), base_test_dataset)]
        base_test_dataset = [*map(lambda x: (base_processor_type.repair(x[0]), x[1]), base_test_dataset)]
        _, full_test_labels = [*map(list, zip(*base_test_dataset))]
        target_labels = [(labels + torch.randint(1, 9, (labels.shape[0],))) % 10 for labels in full_test_labels]
        base_test_dataset = [*map(add_edge_and_tri_offset, base_test_dataset)]
        _, _, full_batched_dataset = gen_adversarial_dataset(base_nn, base_test_dataset, target_labels, batch_size,
                                                             epsilon, targeted, no_epoch=no_epoch)

    return full_batched_dataset, target_labels


def run_transferability_attack(base_nn, target_nn, target_processor_type, full_batched_feature_dct, target_labels):
    target_nn = load_trained_NN(target_nn, dataset, target_processor_type)
    target_test_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, target_processor_type,
                                           AdversarialImageProcessor, 2000, train=False)
    target_test_dataset = DataLoader(target_test_data, batch_size=batch_size, collate_fn=target_processor_type.batch,
                                     num_workers=8, shuffle=False, pin_memory=True)
    target_test_dataset = [data for data in target_test_dataset]
    target_test_dataset = [*map(lambda x: (target_processor_type.clean_features(x[0]), x[1]), target_test_dataset)]
    target_test_dataset = [*map(lambda x: (target_processor_type.repair(x[0]), x[1]), target_test_dataset)]
    target_test_dataset = [*map(add_edge_and_tri_offset, target_test_dataset)]
    print(f"Accuracy of {target_nn.__class__.__name__} before transfer attack from {base_nn.__class__.__name__}")
    run_test_set(target_nn, target_test_dataset, target_labels, batch_size)

    target_feature_dct, full_test_labels = [*map(list, zip(*target_test_dataset))]
    for i in range(len(target_feature_dct)):
        target_feature_dct[i].X0 = full_batched_feature_dct[i].X0
    target_test_dataset = [*zip(target_feature_dct, full_test_labels)]
    print(f"Accuracy of {target_nn.__class__.__name__} after transfer attack from {base_nn.__class__.__name__}")
    run_test_set(target_nn, target_test_dataset, target_labels, batch_size)


if __name__ == "__main__":
    processor_type, NN = model
    if NN in {superpixel_GCN[1], superpixel_GAT[1]}:
        NN = NN(5, output_size).to(DEVICE)
    else:
        NN = NN(5, 10, 15, output_size).to(DEVICE)

    if direct_attack:
        run_direct_attack(processor_type, NN, targeted=targeted)
    else:
        full_batched_dataset, target_labels = None, None
        full_batched_dataset, target_labels = gen_transferability_attack(NN, processor_type,
            epsilon=0.001, targeted=targeted, full_batched_dataset=full_batched_dataset, target_labels=target_labels, no_epoch=50)
        for target_gnn in model_list:
            target_processor_type, target_nn = target_gnn
            if target_nn in {superpixel_GCN[1], superpixel_GAT[1]}:
                target_nn = target_nn(5, output_size).to(DEVICE)
            else:
                target_nn = target_nn(5, 10, 15, output_size).to(DEVICE)
            run_transferability_attack(NN, target_nn, target_processor_type, full_batched_dataset,
                                       target_labels)
            print()
