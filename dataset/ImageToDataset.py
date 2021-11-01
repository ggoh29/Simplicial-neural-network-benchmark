import torch
from skimage.future import graph
from skimage.segmentation import slic
from skimage.measure import regionprops
from utils import triangle_to_edge_matrix, edge_to_node_matrix
from constants import DEVICE
import numpy as np
from multiprocessing import Pool


class ImageToSimplicialComplex:

    def __init__(self, superpixel_size, sc_type, pool_size, simplicial_complex_size=2):
        self.spixel_size = superpixel_size
        self.sc_type = sc_type
        self.sc_size = simplicial_complex_size
        self.p_size = pool_size


    def process_batch(self, images):
        with Pool(self.p_size) as p:
            feature_and_lapacian_list = p.map(self.image_to_lapacian, images)
        X_batch, L_batch, batch_size = self.sanitize_input_for_batch(feature_and_lapacian_list)

        return X_batch, L_batch, batch_size


    def collated_data_to_batch(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        X_batch, L_batch, batch_size = self.sanitize_input_for_batch(graphs)

        return (X_batch, L_batch, batch_size), labels


    def sanitize_input_for_batch(self, input_list):

        X, L_i, L_v = [*zip(*input_list)]
        X, L_i, L_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)]

        X_batch, L_batch, batch_size = self.make_batch(X, L_i, L_v)

        del X, L_v, L_i

        return X_batch, L_batch, batch_size


    def make_batch(self, X, L_i, L_v):

        X_batch, L_batch, batch_size = [], [], []
        for i in range(self.sc_size + 1):
            x, l, batch = self.batch(X[i], L_i[i], L_v[i])
            X_batch.append(x)
            L_batch.append(l)
            batch_size.append(batch)

        return X_batch, L_batch, batch_size


    def batch(self, x_list, L_i_list, l_v_list):
        feature_batch = torch.cat(x_list, dim=0)

        sizes = [*map(lambda x: x.size()[0], x_list)]
        L_i_list = list(L_i_list)
        mx = 0
        for i in range(1, len(sizes)):
            mx += sizes[i - 1]
            L_i_list[i] += mx
        L_cat = torch.cat(L_i_list, dim=1)
        V_cat = torch.cat(l_v_list, dim=0)
        lapacian_batch = torch.sparse_coo_tensor(L_cat, V_cat)
        batch = [[i for _ in range(sizes[i])] for i in range(len(sizes))]
        batch = torch.tensor([i for sublist in batch for i in sublist], device=DEVICE)
        return feature_batch, lapacian_batch, batch


    def image_to_lapacian(self, img):
        image = (img.double().numpy())[0]
        nodes, edges, triangles, node_features = self.image_to_features(image)
        return self.features_to_lapacians(nodes, edges, triangles, node_features)


    def image_to_features(self, image):
        superpixel = slic(image, n_segments=self.spixel_size, compactness=1, start_label=1)
        rag = graph.rag_mean_color(image, superpixel)
        regions = regionprops(superpixel)

        nodes, edges, triangles, node_features = self.sc_type.convert_graph(rag, regions)
        return nodes, edges, triangles, node_features


    def features_to_lapacians(self, nodes, edges, triangles, node_features):

        sigma1 = edge_to_node_matrix(edges, nodes)
        sigma1 = sigma1.to_sparse()

        X0 = torch.tensor(node_features, dtype=torch.float, device=DEVICE)
        L0 = torch.sparse.mm(sigma1, sigma1.t())

        if self.sc_size == 0:
            return [[X0], [L0.coalesce().indices()], [L0.coalesce().values()]]

        X1 = []
        for x, y in edges:
            X1.append(node_features[x - 1] + node_features[y - 1])

        X1 = torch.tensor(X1, dtype=torch.float, device=DEVICE)

        if self.sc_size == 1:
            L1 = torch.sparse.mm(sigma1.t(), sigma1)
            return [[X0, X1], [L0.coalesce().indices(), L1.coalesce().indices()],
                    [L0.coalesce().values(), L1.coalesce().values()]]

        X2 = []
        for i, j, k in triangles:
            X2.append(node_features[i - 1] + node_features[j - 1] + node_features[k - 1])

        X2 = torch.tensor(X2, dtype=torch.float, device=DEVICE)

        sigma2 = triangle_to_edge_matrix(triangles, edges)
        sigma2 = sigma2.to_sparse()

        L1 = torch.sparse.FloatTensor.add(torch.sparse.mm(sigma1.t(), sigma1), torch.sparse.mm(sigma2, sigma2.t()))
        L2 = torch.sparse.mm(sigma2.t(), sigma2)

        # splitting the sparse tensor as pooling cannot return sparse and to make preparation for minibatching easier
        return [[X0, X1, X2],
                [L0.coalesce().indices(), L1.coalesce().indices(), L2.coalesce().indices()],
                [L0.coalesce().values(), L1.coalesce().values(), L2.coalesce().values()]]
