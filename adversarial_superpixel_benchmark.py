import copy
from Superpixel.SuperpixelDataset import SuperpixelSCDataset
from Superpixel.EdgeFlow import PixelBasedEdgeFlow
from Superpixel.ImageProcessor import ImageProcessor, AdversarialImageProcessor
from torch.utils.data import DataLoader
from constants import DEVICE
from models import superpixel_GCN, superpixel_GAT, superpixel_ESNN, superpixel_BSNN, superpixel_SAT, superpixel_SAN
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


def convert_to_device(lst):
    return [i.to(DEVICE) for i in lst]


def load_trained_NN(NN, dataset, processor_type):
    if not os.path.isfile(f'./data/{NN.__class__.__name__}_nn.pkl'):
        train_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, ImageProcessor,
                                         12000, train=True)

        optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        _ = train(NN, 200, train_data, optimizer, criterion, processor_type)

        torch.save(NN.state_dict(), f'./data/{NN.__class__.__name__}_nn.pkl')
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
            training_acc += (train_acc - training_acc) / i
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


def gen_adversarial_dataset(NN, dataloader, full_target_labels, batch_size, epsilon=0.001, targeted=False):
    NN.eval()
    no_epoch = 250
    full_batched_feature_dct, full_test_labels = [*map(list, zip(*dataloader))]

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
            batched_feature_dct, test_labels = full_batched_feature_dct[i], full_test_labels[i]
            target_labels = full_target_labels[i]

            target_labels, test_labels = target_labels.to(DEVICE), test_labels.to(DEVICE)
            feature_dct = copy.deepcopy(batched_feature_dct)
            feature_dct = add_edge_and_tri_features(feature_dct)
            feature_dct = {key: convert_to_device(feature_dct[key]) for key in feature_dct}

            set_grad(feature_dct)
            prediction = NN(feature_dct)
            initial_pred = prediction.argmax(dim=1)
            initial_acc += test_labels.eq(initial_pred).sum().item() / batch_size
            if targeted:
                loss = loss_f(prediction, target_labels)
            else:
                loss = loss_f(prediction, test_labels)

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

            m.append(data_grad)

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
        acc.append(final_acc / start_initial_acc)
        adv_acc /= (i + 1)

        if epoch % 50 == 0:
            print(f"\nInitial accuracy of {start_initial_acc}, final accuracy of {final_acc}, target accuracy of {adv_acc} of {NN.__class__.__name__}")

        m = torch.cat(m, dim=0)
        m = m.cpu()
        m = torch.sum(m.pow(2))
        gradient.append(m.item())


    print(acc)
    print(gradient)
    print(
        f"Initial accuracy of {start_initial_acc}, final accuracy of {final_acc}, target accuracy of {adv_acc} of {NN.__class__.__name__}")
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
    test_dataset = [*map(lambda x: (processor_type.clean_feature_dct(x[0]), x[1]), test_dataset)]
    test_dataset = [*map(lambda x: (processor_type.repair(x[0]), x[1]), test_dataset)]
    _, full_test_labels = [*map(list, zip(*test_dataset))]
    target_labels = [(labels + torch.randint(1, 9, (labels.shape[0],))) % 10 for labels in full_test_labels]
    test_dataset = [*map(add_edge_and_tri_offset, test_dataset)]
    gen_adversarial_dataset(NN, test_dataset, target_labels, batch_size, targeted=targeted)


def gen_transferability_attack(base_nn, base_processor_type, epsilon=0.001, targeted=True):
    """Test the attack transferability of perturbed images in which the adversarial images are generated for base_nn but
    used against target_nn"""
    base_nn = load_trained_NN(base_nn, dataset, base_processor_type)
    base_test_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, base_processor_type,
                                         AdversarialImageProcessor, 2000, train=False)
    base_test_dataset = DataLoader(base_test_data, batch_size=batch_size, collate_fn=base_processor_type.batch,
                                   num_workers=8, shuffle=False, pin_memory=True)
    base_test_dataset = [data for data in base_test_dataset]
    base_test_dataset = [*map(lambda x: (base_processor_type.clean_feature_dct(x[0]), x[1]), base_test_dataset)]
    base_test_dataset = [*map(lambda x: (base_processor_type.repair(x[0]), x[1]), base_test_dataset)]
    _, full_test_labels = [*map(list, zip(*base_test_dataset))]
    target_labels = [(labels + torch.randint(1, 9, (labels.shape[0],))) % 10 for labels in full_test_labels]
    base_test_dataset = [*map(add_edge_and_tri_offset, base_test_dataset)]
    _, _, full_batched_feature_dct = gen_adversarial_dataset(base_nn, base_test_dataset, target_labels, batch_size,
                                                             epsilon, targeted)

    return full_batched_feature_dct, target_labels


def run_transferability_attack(base_nn, target_nn, target_processor_type, full_batched_feature_dct, target_labels):
    target_nn = load_trained_NN(target_nn, dataset, target_processor_type)
    target_test_data = SuperpixelSCDataset('./data', dataset, superpixel_size, edgeFlow, target_processor_type,
                                           AdversarialImageProcessor, 2000, train=False)
    target_test_dataset = DataLoader(target_test_data, batch_size=batch_size, collate_fn=target_processor_type.batch,
                                     num_workers=8, shuffle=False, pin_memory=True)
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
    # NN_list = [superpixel_GCN, superpixel_GAT, superpixel_ESNN, superpixel_BSNN, superpixel_SAT]
    NN_list = [superpixel_SAT]
    for _ in range(1):
        for processor_type, NN in NN_list:
            NN = NN(5, 10, 15, output_size)
            run_direct_attack(processor_type, NN.to(DEVICE))
    # for i in range(1, 6):
    #     base_processor_type, base_nn = superpixel_Bunch_nn
    #     base_nn = base_nn(5, 10, 15, output_size).to(DEVICE)
    #     full_batched_feature_dct, target_labels = gen_transferability_attack(base_nn, base_processor_type, epsilon=0.001 * i, targeted = False)
    #     for target_gnn in [superpixel_gnn, superpixel_gat, superpixel_Ebli_nn, superpixel_sat_nn]:
    #         target_processor_type, target_nn = target_gnn
    #         if target_nn in {superpixel_gnn[1], superpixel_gat[1]}:
    #             target_nn = target_nn(5, output_size).to(DEVICE)
    #         else:
    #             target_nn = target_nn(5, 10, 15, output_size).to(DEVICE)
    #         run_transferability_attack(base_nn, target_nn, target_processor_type, full_batched_feature_dct, target_labels)
    #         print()

# [0.9914698162729659, 0.9803149606299213, 0.9671916010498688, 0.9566929133858267, 0.9507874015748031, 0.9442257217847769, 0.9356955380577429, 0.9212598425196851, 0.9114173228346457, 0.8989501312335959, 0.8871391076115485, 0.8805774278215223, 0.8740157480314961, 0.8635170603674541, 0.8530183727034122, 0.8464566929133858, 0.8366141732283464, 0.8313648293963255, 0.8228346456692913, 0.8143044619422573, 0.8038057742782152, 0.7933070866141733, 0.7847769028871391, 0.7801837270341208, 0.7723097112860893, 0.7637795275590551, 0.7565616797900262, 0.7460629921259843, 0.7368766404199476, 0.730971128608924, 0.7230971128608923, 0.7211286089238844, 0.7158792650918635, 0.7099737532808399, 0.708005249343832, 0.7034120734908137, 0.6975065616797901, 0.6916010498687665, 0.6863517060367454, 0.681758530183727, 0.6692913385826772, 0.6620734908136483, 0.6528871391076116, 0.6456692913385826, 0.639763779527559, 0.6318897637795277, 0.6259842519685039, 0.6213910761154856, 0.6135170603674541, 0.6082677165354331, 0.6030183727034121, 0.5984251968503937, 0.5918635170603675, 0.589238845144357, 0.5826771653543307, 0.5761154855643045, 0.5702099737532809, 0.5662729658792651, 0.5610236220472441, 0.5570866141732284, 0.5544619422572179, 0.5485564304461943, 0.5433070866141733, 0.5413385826771654, 0.5374015748031497, 0.5341207349081365, 0.5255905511811023, 0.5236220472440944, 0.5190288713910761, 0.515748031496063, 0.5078740157480316, 0.5032808398950132, 0.49934383202099736, 0.49868766404199477, 0.49475065616797903, 0.49278215223097116, 0.4881889763779528, 0.4862204724409449, 0.4809711286089239, 0.47834645669291337, 0.47506561679790027, 0.4698162729658793, 0.4665354330708662, 0.4625984251968504, 0.45866141732283466, 0.4547244094488189, 0.45013123359580054, 0.44291338582677164, 0.43963254593175854, 0.43503937007874016, 0.43175853018372706, 0.42585301837270345, 0.42191601049868765, 0.42191601049868765, 0.41929133858267714, 0.4146981627296588, 0.4107611548556431, 0.40813648293963256, 0.40748031496063, 0.40682414698162733, 0.4041994750656168, 0.40157480314960625, 0.39763779527559057, 0.3937007874015748, 0.3937007874015748, 0.3904199475065617, 0.38845144356955386, 0.38582677165354334, 0.38254593175853024, 0.3812335958005249, 0.37926509186351703, 0.3746719160104987, 0.37270341207349084, 0.3700787401574803, 0.36811023622047245, 0.3661417322834646, 0.36220472440944884, 0.36154855643044614, 0.35826771653543305, 0.3556430446194226, 0.35236220472440943, 0.35236220472440943, 0.349737532808399, 0.3477690288713911, 0.3451443569553806, 0.344488188976378, 0.34251968503937014, 0.3418635170603675, 0.3379265091863517, 0.3353018372703412, 0.33136482939632544, 0.32808398950131235, 0.32742782152230976, 0.3241469816272966, 0.32217847769028873, 0.3215223097112861, 0.3195538057742782, 0.31758530183727035, 0.31692913385826776, 0.3143044619422572, 0.3123359580052493, 0.30708661417322836, 0.30249343832021, 0.30183727034120733, 0.2992125984251969, 0.29855643044619423, 0.2946194225721785, 0.291994750656168, 0.28937007874015747, 0.28608923884514437, 0.28477690288713914, 0.2841207349081365, 0.2828083989501312, 0.281496062992126, 0.28018372703412076, 0.2782152230971129, 0.276246719160105, 0.2749343832020998, 0.2736220472440945, 0.27165354330708663, 0.27034120734908135, 0.26968503937007876, 0.2683727034120735, 0.2664041994750656, 0.265748031496063, 0.26377952755905515, 0.2618110236220472, 0.25918635170603677, 0.2578740157480315, 0.255249343832021, 0.2539370078740158, 0.2545931758530184, 0.25131233595800523, 0.25131233595800523, 0.24803149606299213, 0.2467191601049869, 0.24606299212598426, 0.24540682414698164, 0.24540682414698164, 0.2440944881889764, 0.2427821522309711, 0.2427821522309711, 0.2421259842519685, 0.24146981627296588, 0.2388451443569554, 0.23818897637795275, 0.23556430446194226, 0.2335958005249344, 0.2335958005249344, 0.23293963254593178, 0.23293963254593178, 0.23097112860892388, 0.229002624671916, 0.2283464566929134, 0.22637795275590553, 0.22440944881889766, 0.2230971128608924, 0.22178477690288712, 0.21850393700787402, 0.2178477690288714, 0.21587926509186353, 0.21456692913385828, 0.2119422572178478, 0.2106299212598425, 0.20866141732283464, 0.20800524934383202, 0.20603674540682415, 0.20538057742782154, 0.20603674540682415, 0.20472440944881892, 0.20275590551181105, 0.2020997375328084, 0.20144356955380577, 0.20078740157480313, 0.2001312335958005, 0.19750656167979003, 0.19619422572178477, 0.19619422572178477, 0.19422572178477693, 0.19356955380577429, 0.19028871391076113, 0.19028871391076113, 0.1883202099737533, 0.18766404199475065, 0.18569553805774278, 0.18503937007874016, 0.1830708661417323, 0.18241469816272968, 0.18044619422572178, 0.17913385826771652, 0.1784776902887139, 0.17716535433070865, 0.17650918635170604, 0.17519685039370078, 0.17454068241469817, 0.1725721784776903, 0.1725721784776903, 0.17060367454068243, 0.17060367454068243, 0.16929133858267717, 0.16994750656167978, 0.16929133858267717, 0.1679790026246719, 0.1679790026246719, 0.1679790026246719, 0.1673228346456693, 0.16666666666666666, 0.16601049868766404, 0.16404199475065617, 0.16338582677165356]
# [21.010448455810547, 18.842422485351562, 20.797765731811523, 24.441743850708008, 18.57802391052246, 20.699586868286133, 20.770095825195312, 19.899063110351562, 25.722190856933594, 25.34467315673828, 31.81755828857422, 21.720447540283203, 18.754989624023438, 20.655147552490234, 20.97234535217285, 21.33929443359375, 24.1907958984375, 24.499370574951172, 23.974063873291016, 23.774404525756836, 21.906539916992188, 24.248624801635742, 22.635772705078125, 22.099824905395508, 20.87874984741211, 21.895368576049805, 20.097997665405273, 22.846149444580078, 21.37139320373535, 23.577083587646484, 25.630033493041992, 24.67828369140625, 23.836956024169922, 21.31707000732422, 20.456520080566406, 21.17766761779785, 23.20375633239746, 22.73569679260254, 27.076004028320312, 26.060279846191406, 24.713111877441406, 25.29254150390625, 28.183876037597656, 25.624937057495117, 25.0373592376709, 24.77553367614746, 27.876310348510742, 25.009868621826172, 24.72640037536621, 28.32726287841797, 24.94955825805664, 27.73224449157715, 23.808067321777344, 28.863758087158203, 23.225236892700195, 27.77312660217285, 24.140493392944336, 27.683807373046875, 22.033794403076172, 26.24539566040039, 22.29449462890625, 24.17840576171875, 23.152021408081055, 22.396196365356445, 21.823894500732422, 20.75505256652832, 23.693571090698242, 19.153841018676758, 22.938087463378906, 19.548830032348633, 25.697179794311523, 18.129953384399414, 22.357421875, 19.005062103271484, 23.590970993041992, 20.170021057128906, 22.225109100341797, 20.404666900634766, 22.30576515197754, 19.045469284057617, 23.356136322021484, 18.3779296875, 22.896183013916016, 20.283830642700195, 23.067771911621094, 21.723094940185547, 23.988018035888672, 21.178674697875977, 23.454570770263672, 27.603649139404297, 31.94242286682129, 23.49630355834961, 23.821062088012695, 19.878402709960938, 24.819854736328125, 21.657997131347656, 23.66722869873047, 23.278026580810547, 21.351879119873047, 24.062667846679688, 21.623653411865234, 23.61570167541504, 22.097694396972656, 23.376726150512695, 21.185449600219727, 23.36761474609375, 21.65330696105957, 22.7720890045166, 21.86768341064453, 20.32794761657715, 23.52528953552246, 20.589435577392578, 24.608312606811523, 19.88027000427246, 26.72275733947754, 18.495031356811523, 25.80140495300293, 21.761409759521484, 28.745121002197266, 17.345947265625, 28.86297607421875, 19.79184341430664, 32.90061950683594, 19.67210578918457, 30.21198844909668, 18.313352584838867, 30.540447235107422, 15.515345573425293, 29.662883758544922, 16.03348159790039, 27.320232391357422, 15.716848373413086, 25.62091636657715, 15.320472717285156, 21.94192886352539, 14.599994659423828, 22.430233001708984, 14.393157958984375, 23.77275276184082, 13.973981857299805, 24.61569595336914, 14.377036094665527, 22.783048629760742, 14.59647274017334, 21.753686904907227, 14.248791694641113, 20.169248580932617, 16.164966583251953, 21.215307235717773, 16.828594207763672, 21.287927627563477, 17.77754020690918, 20.39505958557129, 16.95980453491211, 20.195131301879883, 15.723751068115234, 19.50664520263672, 17.222158432006836, 20.131793975830078, 15.032938003540039, 20.346355438232422, 15.545180320739746, 22.379581451416016, 16.322145462036133, 21.28472900390625, 15.193459510803223, 20.672508239746094, 15.143261909484863, 20.339553833007812, 14.487608909606934, 20.71803092956543, 14.880671501159668, 20.1220760345459, 14.213481903076172, 20.531166076660156, 14.386428833007812, 20.722002029418945, 14.770844459533691, 22.88886260986328, 15.541189193725586, 21.424991607666016, 15.548483848571777, 21.558490753173828, 15.225787162780762, 21.97711944580078, 15.894895553588867, 22.2421875, 17.16337013244629, 22.37773895263672, 16.989078521728516, 22.462276458740234, 17.728084564208984, 22.91116714477539, 17.43226432800293, 23.2043514251709, 17.785926818847656, 24.181957244873047, 18.11048126220703, 24.378944396972656, 18.55592155456543, 24.55056381225586, 18.672950744628906, 25.91916847229004, 18.844905853271484, 25.173709869384766, 18.596031188964844, 25.261306762695312, 18.461551666259766, 26.08064079284668, 17.3807373046875, 25.120346069335938, 21.68590545654297, 22.94223976135254, 21.186857223510742, 21.86611557006836, 20.39934730529785, 23.340862274169922, 20.124309539794922, 22.74618148803711, 20.563350677490234, 23.54969024658203, 19.94450569152832, 22.161619186401367, 19.106534957885742, 22.0968017578125, 18.801734924316406, 23.151220321655273, 20.671852111816406, 21.506587982177734, 18.55519676208496, 21.197242736816406, 18.37192726135254, 20.56027603149414, 18.547889709472656, 20.164772033691406, 18.03982925415039, 21.0595645904541, 18.12190055847168, 21.23347282409668, 17.531116485595703, 21.181705474853516, 17.730121612548828, 20.65758514404297, 17.333173751831055, 19.715103149414062, 17.017229080200195, 18.469772338867188, 16.747400283813477, 18.26300811767578, 16.376346588134766]
# Initial accuracy of 0.7559523809523809, final accuracy of 0.12351190476190477, target accuracy of 0.09821428571428571 of SuperpixelSAT
