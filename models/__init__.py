from models.GNN.model import SuperpixelGCN, SuperpixelGAT, PlanetoidGAT, PlanetoidGCN
from models.ESNN.model import SuperpixelEbli, FlowEbli
from models.BSNN.model import SuperpixelBunch, PlanetoidBunch, FlowBunch
from models.SAT.model import SuperpixelSAT, FlowSAT
from models.SAN.model import FlowSAN

from models.GNN.GNNProcessor import GNNProcessor
from models.ESNN.ESNNProcessor import ESNNProcessor
from models.BSNN.BSNNProcessor import BSNNProcessor
from models.SAT.SATProcessor import SATProcessor


superpixel_ESNN = [ESNNProcessor(), SuperpixelEbli]
superpixel_BSNN = [BSNNProcessor(), SuperpixelBunch]
superpixel_SAT = [SATProcessor(), SuperpixelSAT]
superpixel_GCN = [GNNProcessor(), SuperpixelGCN]
superpixel_GAT = [GNNProcessor(), SuperpixelGAT]

planetoid_BSNN = [BSNNProcessor(), PlanetoidBunch]
planetoid_GCN = [GNNProcessor(), PlanetoidGCN]
planetoid_GAT = [GNNProcessor(), PlanetoidGAT]

flow_SAT = [SATProcessor(), FlowSAT]
flow_SAN = [SATProcessor(), FlowSAN]
flow_ESNN = [ESNNProcessor(), FlowEbli]
flow_BSNN = [BSNNProcessor(), FlowBunch]