from Planetoid.PlanetoidDataset.PlanetoidLoader import PlanetoidSCDataset
from models.all_models import planetoid_gat, planetoid_gnn, planetoid_Bunch_nn, planetoid_Ebli_nn, planetoid_sat
import torch.nn as nn
import torch
from Planetoid.DGI.DGI import DGI
from Planetoid.DGI.logreg import LogReg
from constants import DEVICE

dataset = 'Cora'
dataset_features_dct = {'Cora' : 1433, 'CiteSeer' : 3703, 'PubMed' : 500}
dataset_classes_dct = {'Cora' : 7, 'CiteSeer' : 6, 'PubMed' : 3}
input_size = dataset_features_dct[dataset]
output_size = 512
nb_epochs = 500
test_epochs = 50
lr = 0.001
l2_coef = 0.0
patience = 20

# nn_mod = planetoid_gnn
nn_mod = planetoid_Bunch_nn
# nn_mod = planetoid_sat
processor_type = nn_mod[0]
model = nn_mod[1]

model = DGI(input_size, output_size, model)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if __name__ == "__main__":

    data = PlanetoidSCDataset('./data', dataset, processor_type)
    data_dct = data.get_full()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    thing = data_dct['lapacian'][0].to_dense().clone()
    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        nb_nodes = data_dct["features"][0].shape[0]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)

        lbl = torch.cat((lbl_1, lbl_2), 1).to(DEVICE)

        logits = model(data_dct, processor_type)

        loss = b_xent(logits, lbl)

        print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))

    embeds, _ = model.embed(data_dct)

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
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()

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
        print(acc)
        tot += acc

    print('Average accuracy:', tot / test_epochs)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())