from Planetoid.PlanetoidDataset.PlanetoidLoader import PlanetoidSCDataset
from models.GNN.GNNProcessor import GNNProcessor
from models.nn_utils import unpack_feature_dct_to_L_X_B
from Planetoid.PlanetoidDataset.PlanetoidLoader import convert_to_SC
import numpy as np
import torch
from constants import DEVICE

processor_type = GNNProcessor()


if __name__ == "__main__":

    data = PlanetoidSCDataset('./data', 'Cora', processor_type)
    train_dct = processor_type.batch([data[0]])[0]
    train_dct = processor_type.clean_feature_dct(train_dct)

