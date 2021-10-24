import skimage
from skimage import data, io, segmentation, color, draw, filters
import numpy as np
from functools import reduce
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


    # def prepareForMiniBatching(self, x_list, L_list):
    #
    #     feature_batch = torch.cat(x_list, dim=0)
    #
    #     sizes = [*map(lambda x : x.size()[0], x_list)]
    #     L = [l.coalesce().indices() for l in L_list]
    #     mx = 0
    #     for i in range(1, len(sizes)):
    #         mx += sizes[i-1]
    #         L[i] += mx
    #     V = [l.coalesce().values() for l in L_list]
    #     L_cat = torch.cat(L, dim = 1)
    #     V_cat = torch.cat(V, dim = 0)
    #     lapacian_batch = torch.sparse_coo_tensor(L_cat, V_cat)
    #     batch = [[i for _ in range(sizes[i])] for i in range(len(sizes))]
    #     batch = torch.tensor([i for sublist in batch for i in sublist], device=device)
    #     return feature_batch, lapacian_batch, batch
    #
    # def process_batch_and_feed_to_NN(self, NN, images, p):
    #     feature_and_lapacian_list = [*map(self.getFeaturesAndLapacians, images)]
    #     X0, X1, X2, L0, L1, L2 = [*zip(*feature_and_lapacian_list)]
    #     X0, L0, batch0 = self.prepareForMiniBatching(X0, L0)
    #     X1, L1, batch1 = self.prepareForMiniBatching(X1, L1)
    #     X2, L2, batch2 = self.prepareForMiniBatching(X2, L2)
    #     return NN(X0, X1, X2, L0, L1, L2, batch0, batch1, batch2)

    def process_batch_and_feed_to_NN(self, NN, images, p):
        X0, X1, X2, L0, L1, L2 = self.getFeaturesAndLapacians(images[0])
        batch0, batch1, batch2 = 0,0,0

        return NN(X0, X1, X2, L0, L1, L2, batch0, batch1, batch2)
