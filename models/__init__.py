from models.GNN.model import SuperpixelGCN, SuperpixelGAT, PlanetoidGAT, PlanetoidGCN
from models.ESNN.model import SuperpixelEbli, PlanetoidEbli, FlowEbli, TestEbli
from models.BSNN.model import SuperpixelBunch, PlanetoidBunch, FlowBunch, TestBunch
from models.SAT.model import SuperpixelSAT, PlanetoidSAT, FlowSAT, TestSAT
from models.SAN.model import SuperpixelSAN, PlanetoidSAN, FlowSAN, TestSAN

from models.GNN.GNNProcessor import GNNProcessor
from models.ESNN.ESNNProcessor import ESNNProcessor
from models.BSNN.BSNNProcessor import BSNNProcessor
from models.SAT.SATProcessor import SATProcessor
from models.SAN.SANProcessor import SANProcessor


superpixel_ESNN = [ESNNProcessor(), SuperpixelEbli]
superpixel_BSNN = [BSNNProcessor(), SuperpixelBunch]
superpixel_SAT = [SATProcessor(), SuperpixelSAT]
superpixel_SAN = [SANProcessor(), SuperpixelSAN]
superpixel_GCN = [GNNProcessor(), SuperpixelGCN]
superpixel_GAT = [GNNProcessor(), SuperpixelGAT]

planetoid_ESNN = [ESNNProcessor(), PlanetoidEbli]
planetoid_BSNN = [BSNNProcessor(), PlanetoidBunch]
planetoid_GCN = [GNNProcessor(), PlanetoidGCN]
planetoid_GAT = [GNNProcessor(), PlanetoidGAT]
planetoid_SAT = [SATProcessor(), PlanetoidSAT]
planetoid_SAN = [SANProcessor(), PlanetoidSAN]

flow_SAT = [SATProcessor(), FlowSAT]
flow_SAN = [SANProcessor(), FlowSAN]
flow_ESNN = [ESNNProcessor(), FlowEbli]
flow_BSNN = [BSNNProcessor(), FlowBunch]

test_SAT = [SATProcessor(), TestSAT]
test_SAN = [SANProcessor(), TestSAN]
test_ESNN = [ESNNProcessor(), TestEbli]
test_BSNN = [BSNNProcessor(), TestBunch]