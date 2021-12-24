from models.GNN.model import SuperpixelGCN, SuperpixelGAT
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.model_Ebli import Superpixel_Ebli
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.model_Bunch import Superpixel_Bunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from models.SAT.model import Superpixel_SAT
from models.SAT.SATProcessor import SATProcessor


Ebli_nn = [SNNEbliProcessor(), Superpixel_Ebli]
Bunch_nn = [SNNBunchProcessor(), Superpixel_Bunch]
sat_nn = [SATProcessor(), Superpixel_SAT]
gnn = [GNNProcessor(), SuperpixelGCN]
gat = [GNNProcessor(), SuperpixelGAT]
gcn_list = [gnn, gat]

