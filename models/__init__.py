from models.GNN.model import SuperpixelGCN, SuperpixelGAT, PlanetoidGAT, PlanetoidGCN
from models.GNN.GNNProcessor import GNNProcessor
from models.ESNN.model import SuperpixelEbli
from models.ESNN.ESNNProcessor import ESNNProcessor
from models.BSNN.model import SuperpixelBunch, PlanetoidBunch
from models.BSNN.BSNNProcessor import BSNNProcessor
from models.SAT.model import SuperpixelSAT
from models.SAT.SATProcessor import SATProcessor

superpixel_ESNN = [ESNNProcessor(), SuperpixelEbli]
superpixel_BSNN = [BSNNProcessor(), SuperpixelBunch]
superpixel_SAT = [SATProcessor(), SuperpixelSAT]
superpixel_GCN = [GNNProcessor(), SuperpixelGCN]
superpixel_GAT = [GNNProcessor(), SuperpixelGAT]

planetoid_BSNN = [BSNNProcessor(), PlanetoidBunch]
planetoid_GCN = [GNNProcessor(), PlanetoidGCN]
planetoid_GAT = [GNNProcessor(), PlanetoidGAT]