from models.GNN.model import SuperpixelGCN, SuperpixelGAT, PlanetoidGAT, PLanetoidGCN
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.model_Ebli import SuperpixelEbli
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.model_Bunch import SuperpixelBunch, PlanetoidBunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from models.SAT.model import SuperpixelSAT
from models.SAT.SATProcessor import SATProcessor

superpixel_Ebli_nn = [SNNEbliProcessor(), SuperpixelEbli]
superpixel_Bunch_nn = [SNNBunchProcessor(), SuperpixelBunch]
superpixel_sat_nn = [SATProcessor(), SuperpixelSAT]
superpixel_gnn = [GNNProcessor(), SuperpixelGCN]
superpixel_gat = [GNNProcessor(), SuperpixelGAT]

planetoid_Bunch_nn = [SNNBunchProcessor(), PlanetoidBunch]
planetoid_gnn = [GNNProcessor(), PLanetoidGCN]
planetoid_gat = [GNNProcessor(), PlanetoidGAT]