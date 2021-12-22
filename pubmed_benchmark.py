from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

if __name__ == "__main__":
    data = Planetoid('./data', 'PubMed')
    print(data[0].val_mask)
