from models.GNN.model import SuperpixelGCN, SuperpixelGAT
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.model_Ebli import SuperpixelEbli
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.model_Bunch import SuperpixelBunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from models.SAT.model import SuperpixelSAT
from models.SAT.SATProcessor import SATProcessor


Ebli_nn = [SNNEbliProcessor(), SuperpixelEbli]
Bunch_nn = [SNNBunchProcessor(), SuperpixelBunch]
sat_nn = [SATProcessor(), SuperpixelSAT]
gnn = [GNNProcessor(), SuperpixelGCN]
gat = [GNNProcessor(), SuperpixelGAT]

