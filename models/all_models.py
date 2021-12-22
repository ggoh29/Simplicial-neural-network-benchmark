from models.GNN.model import GCN, GAT
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.model_E import SNN_Ebli
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.model_B import SNN_Bunch
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from models.SAT.model import SAT
from models.SAT.SATProcessor import SATProcessor


Ebli_nn = [SNNEbliProcessor(), SNN_Ebli]
Bunch_nn = [SNNBunchProcessor(), SNN_Bunch]
sat_nn = [SATProcessor(), SAT]
snn_list = [Ebli_nn, Bunch_nn, sat_nn]
gnn = [GNNProcessor(), GCN]
gat = [GNNProcessor(), GAT]
gcn_list = [gnn, gat]

