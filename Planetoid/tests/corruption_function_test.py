import unittest
from Planetoid.PlanetoidDataset.PlanetoidLoader import PlanetoidSCDataset
from models import planetoid_gat, planetoid_gnn, planetoid_Bunch_nn, planetoid_Ebli_nn
from Planetoid.DGI.DGI import corruption_function
import torch.nn as nn
import torch
from Planetoid.DGI.DGI import DGI
from Planetoid.DGI.logreg import LogReg
from constants import DEVICE


class MyTestCase(unittest.TestCase):
    def test_corruption_function_with_p_set_to_0_does_not_change_lapacian(self):
        dataset = 'Cora'

        nn_mod = planetoid_gnn
        processor_type = nn_mod[0]

        data = PlanetoidSCDataset('./data', dataset, processor_type)
        data_dct = data.get_full()
        corrupted_dct = corruption_function(data_dct, processor_type, p = 0)

        self.assertTrue(torch.allclose(data_dct['features'][0], corrupted_dct['features'][0], atol=1e-5))
        self.assertTrue(torch.allclose(data_dct['lapacian'][0].to_dense(), corrupted_dct['lapacian'][0].to_dense(), atol=1e-5))
        # self.assertTrue(torch.allclose(data_dct['lapacian'][1].to_dense(), corrupted_dct['lapacian'][1].to_dense(), atol=1e-5))
        # self.assertTrue(torch.allclose(data_dct['lapacian'][2].to_dense(), corrupted_dct['lapacian'][2].to_dense(), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
