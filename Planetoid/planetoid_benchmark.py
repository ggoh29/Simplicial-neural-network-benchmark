from Planetoid.PlanetoidDataset.PlanetoidLoader import PlanetoidSCDataset
from models.all_models import planetoid_gat, planetoid_gnn, planetoid_Bunch_nn, planetoid_Ebli_nn
import torch.nn as nn
import numpy as np
import torch
from Planetoid.DGI.DGI import DGI
from constants import DEVICE

input_size = 1433
output_size = 128
nb_epochs = 5000
lr = 0.001
l2_coef = 0.0
patience = 20

# nn_mod = planetoid_gnn
nn_mod = planetoid_Bunch_nn
processor_type = nn_mod[0]
model = nn_mod[1]


model = DGI(input_size, output_size, model)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if __name__ == "__main__":

    data = PlanetoidSCDataset('./data', 'Cora', processor_type)
    train_dct = processor_type.batch([data[0]])[0]
    train_dct = processor_type.clean_feature_dct(train_dct)
    train_dct = processor_type.repair(train_dct)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        nb_nodes = train_dct["features"][0].shape[0]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(DEVICE)

        logits = model(train_dct, processor_type)

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

