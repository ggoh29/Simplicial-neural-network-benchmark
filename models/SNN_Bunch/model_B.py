import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from models.nn_utils import normalise, \
    convert_indices_and_values_to_sparse, torch_sparse_to_scipy_sparse
from scipy import sparse
import numpy as np

class SNN_Bunch(nn.Module):
  
    def __init__(self, f1_size, f2_size, f3_size, output_size):
        super().__init__()

    def forward(self, features_dct):
      return


