from Planetoid.PlanetoidDataset.PlanetoidLoader import PlanetoidSCDataset
from models.GNN.GNNProcessor import GNNProcessor

processor_type = GNNProcessor()
if __name__ == "__main__":

    data = PlanetoidSCDataset('./root', 'Cora', processor_type)
    print(data[2].X0.shape)
