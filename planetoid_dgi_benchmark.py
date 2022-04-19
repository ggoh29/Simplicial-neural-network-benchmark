from Planetoid.PlanetoidDataset import PlanetoidSCDataset
from models import planetoid_GCN, planetoid_GAT, planetoid_ESNN, planetoid_BSNN, planetoid_SAN, planetoid_SAT
import torch.nn as nn
import torch
from Planetoid.DGI import DGI
from Planetoid.logreg import LogReg
from constants import DEVICE

2708, 79
dataset = 'Cora'
dataset_features_dct = {'Cora' : 1433, 'CiteSeer' : 3703, 'PubMed' : 500, 'fake' : 2708}
dataset_classes_dct = {'Cora' : 7, 'CiteSeer' : 6, 'PubMed' : 3 , 'fake' : 3}
input_size = dataset_features_dct[dataset]
output_size = 512
nb_epochs = 200
test_epochs = 50
lr = 0.001
l2_coef = 0.0
patience = 20

# nn_mod = planetoid_GCN
# nn_mod = planetoid_GAT
# nn_mod = planetoid_ESNN
nn_mod = planetoid_BSNN
# nn_mod = planetoid_SAT
# nn_mod = planetoid_SAN


processor_type = nn_mod[0]
model = nn_mod[1]

dgi = DGI(input_size, output_size, model)
optimiser = torch.optim.Adam(dgi.parameters(), lr=lr, weight_decay=l2_coef)
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()

if __name__ == "__main__":

    data = PlanetoidSCDataset('./data', dataset, processor_type)
    data_full, b1, b2 = data.get_full()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    bl = False
    b1 = b1.to(DEVICE)
    b2 = b2.to(DEVICE)
    for epoch in range(nb_epochs):
        dgi.train()
        optimiser.zero_grad()

        nb_nodes = data_full.X0.shape[0]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)

        lbl = torch.cat((lbl_1, lbl_2), 1).to(DEVICE)

        logits = dgi(data_full, b1, b2, processor_type)

        loss = b_xent(logits, lbl)

        print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), f'./data/{model.__name__}_dgi.pkl')
            if epoch != 0:
                bl = True
        else:
            if bl:
                cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    dgi.load_state_dict(torch.load(f'./data/{model.__name__}_dgi.pkl'))

    embeds, _ = dgi.embed(data_full, b1, b2)
    # embeds = data_full.X0.to(DEVICE)
    # output_size = 79
    # with open("./embeddings.py", 'w') as f:
    #     f.write(f'embeddings = {embeds.tolist()}')
    # with open("./labels.py", 'w') as f:
    #     f.write(f'labels = {data.get_labels().tolist()}')
    train_embs = data.get_train_embeds(embeds)
    val_embs = data.get_val_embeds(embeds)
    test_embs = data.get_test_embeds(embeds)

    train_lbls = data.get_train_labels().to(DEVICE)
    x_unique = train_lbls.unique(sorted=True)
    x_unique_count = torch.stack([(train_lbls == x_u).sum() for x_u in x_unique])
    val_lbls = data.get_val_labels().to(DEVICE)
    test_lbls = data.get_test_labels().to(DEVICE)

    tot = torch.zeros(1).to(DEVICE)

    accs = []

    for _ in range(test_epochs):
        log = LogReg(output_size, dataset_classes_dct[dataset])
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(DEVICE)

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.to(DEVICE)

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(model.__name__)
        print(acc)
        tot += acc

    print('Average accuracy:', tot / test_epochs)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())
