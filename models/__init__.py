from models.GNN.model import SuperpixelGCN, SuperpixelGAT, PlanetoidGAT, PlanetoidGCN
from models.SCN.model import SuperpixelSCN, PlanetoidSCN, FlowSCN, TestSCN
from models.SCConv.model import SuperpixelSCConv, PlanetoidSCConv, FlowSCConv, TestSCConv
from models.SAT.model import SuperpixelSAT, PlanetoidSAT, FlowSAT, TestSAT
from models.SAN.model import SuperpixelSAN, PlanetoidSAN, FlowSAN, TestSAN

from models.GNN.GNNProcessor import GNNProcessor
from models.SCN.SCNProcessor import SCNProcessor
from models.SCConv.SCConvProcessor import SCConvProcessor
from models.SAT.SATProcessor import SATProcessor
from models.SAN.SANProcessor import SANProcessor


superpixel_SCN = [SCNProcessor(), SuperpixelSCN]
superpixel_SCConv = [SCConvProcessor(), SuperpixelSCConv]
superpixel_SAT = [SATProcessor(), SuperpixelSAT]
superpixel_SAN = [SANProcessor(), SuperpixelSAN]
superpixel_GCN = [GNNProcessor(), SuperpixelGCN]
superpixel_GAT = [GNNProcessor(), SuperpixelGAT]

planetoid_SCN = [SCNProcessor(), PlanetoidSCN]
planetoid_SCConv = [SCConvProcessor(), PlanetoidSCConv]
planetoid_GCN = [GNNProcessor(), PlanetoidGCN]
planetoid_GAT = [GNNProcessor(), PlanetoidGAT]
planetoid_SAT = [SATProcessor(), PlanetoidSAT]
planetoid_SAN = [SANProcessor(), PlanetoidSAN]

flow_SAT = [SATProcessor(), FlowSAT]
flow_SAN = [SANProcessor(), FlowSAN]
flow_SCN = [SCNProcessor(), FlowSCN]
flow_SCConv = [SCConvProcessor(), FlowSCConv]

test_SAT = [SATProcessor(), TestSAT]
test_SAN = [SANProcessor(), TestSAN]
test_SCN = [SCNProcessor(), TestSCN]
test_SCConv = [SCConvProcessor(), TestSCConv]