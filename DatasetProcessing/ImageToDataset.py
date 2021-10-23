import skimage
from skimage import data, io, segmentation, color, draw, filters
import numpy as np

import torch
from skimage.future import graph
from skimage.segmentation import slic
from skimage.measure import regionprops
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageToGraph:

    def __init__(self, superpixel_size):
        self.spixel_size = superpixel_size

    def unpack_node(self, rag, region):
        return list(region['centroid']) + list(rag._node[region['label']]['mean color'])


    def getFeaturesAndLapacians(self, img):
        image = (img.double().numpy())[0]
        superpixel = slic(image, n_segments=self.spixel_size, compactness=0.75, start_label = 1)
        rag = graph.rag_mean_color(image, superpixel)
        regions = regionprops(superpixel)
        node_features = [self.unpack_node(rag, region) for region in regions]
        triangles = [*filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(rag))]

        sigma1 = torch.tensor([[0 for _ in rag.edges] for _ in rag.nodes], dtype=torch.float, device = device)
        X1 = []
        j = 0
        for edge in rag.edges():

            x, y = edge
            sigma1[x - 1][j] = -1
            sigma1[y - 1][j] = 1
            j += 1

            X1.append(node_features[x-1] + node_features[y-1])

        sigma2 = torch.tensor([[0 for _ in triangles] for _ in rag.edges], dtype=torch.float, device = device)
        X2 = []
        edges = [e for e in rag.edges]
        for l in range(len(triangles)):
            i, j, k = triangles[l]
            sigma2[edges.index((i, j))][l] = 1
            sigma2[edges.index((j, k))][l] = 1
            sigma2[edges.index((i, k))][l] = -1

            X2.append(node_features[i - 1] + node_features[j - 1] + node_features[k - 1])

        X0, X1, X2 = torch.tensor(node_features,dtype=torch.float, device = device), \
                     torch.tensor(X1,dtype=torch.float, device = device), \
                     torch.tensor(X2, dtype=torch.float, device = device)

        sigma1, sigma2 = sigma1.to_sparse(), sigma2.to_sparse()

        L0 = torch.sparse.mm(sigma1, sigma1.t())
        L1 = torch.sparse.FloatTensor.add(torch.sparse.mm(sigma1.t(), sigma1), torch.sparse.mm(sigma2, sigma2.t()))
        L2 = torch.sparse.mm(sigma2.t(), sigma2)

        return X0, X1, X2, L0, L1, L2

