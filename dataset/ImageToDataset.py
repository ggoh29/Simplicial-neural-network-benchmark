import torch
from skimage.future import graph
from skimage.segmentation import slic
from skimage.measure import regionprops
from utils import triangle_to_edge_matrix, edge_to_node_matrix
from constants import DEVICE
import numpy as np
from multiprocessing import Pool
from utils import dense_to_tensor, tensor_to_dense


def stl(t):
    "Shape to list"
    return list(t.shape)

class SCData:

    def __init__(self, X0, X1, X2, sigma1, sigma2, label):
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2
        # sigma1 and sigma2 can either be sparse or dense but since python doesn't really have overloading, doing this instead
        if sigma1.is_sparse:
            sigma1 = dense_to_tensor(sigma1)
        self.sigma1 = sigma1

        if sigma2.is_sparse:
            sigma2 = dense_to_tensor(sigma2)
        self.sigma2 = sigma2

        self.label = label


    def __str__(self):
        name = f"SCData(X0={stl(self.X0)}, X1={stl(self.X1)}, X2={stl(self.X2)}," \
               f" sigma1={stl(self.sigma1)}, sigma2={stl(self.sigma2)}, label={stl(self.label)})"
        return name

    def __repr__(self):
        name = f"SCData(X0={stl(self.X0)}, X1={stl(self.X1)}, X2={stl(self.X2)}," \
               f" sigma1={stl(self.sigma1)}, sigma2={stl(self.sigma2)}, label={stl(self.label)})"
        return name


class ProcessImage:

    def __init__(self, superpixel_size, edgeflow):
        self.spixel_size = superpixel_size
        self.edgeflow = edgeflow

    def image_to_features(self, image):
        image, label = image
        image = (image.double().numpy())[0]
        label = torch.tensor([label], device=DEVICE)

        superpixel = slic(image, n_segments=self.spixel_size, compactness=1, start_label=1)
        rag = graph.rag_mean_color(image, superpixel)
        regions = regionprops(superpixel)

        nodes, edges, triangles, node_features = self.edgeflow.convert_graph(rag, regions)

        X0 = torch.tensor(node_features, dtype=torch.float, device=DEVICE)

        X1 = []
        for x, y in edges:
            X1.append(node_features[x - 1] + node_features[y - 1])

        X1 = torch.tensor(X1, dtype=torch.float, device=DEVICE)

        X2 = []
        for i, j, k in triangles:
            X2.append(node_features[i - 1] + node_features[j - 1] + node_features[k - 1])

        X2 = torch.tensor(X2, dtype=torch.float, device=DEVICE)

        sigma1 = edge_to_node_matrix(edges, nodes)
        sigma1 = sigma1.to_sparse()

        sigma2 = triangle_to_edge_matrix(triangles, edges)
        sigma2 = sigma2.to_sparse()

        return SCData(X0, X1, X2, sigma1, sigma2, label)


class DatasetBatcher:

    def __init__(self, simplicial_complex_size=2):
        self.sc_size = simplicial_complex_size


    def sc_data_to_lapacian(self, scData):

        def to_sparse_coo(matrix):
            indices = matrix[0:2]
            values = matrix[2:3].squeeze()
            return torch.sparse_coo_tensor(indices, values)

        sigma1, sigma2 = to_sparse_coo(scData.sigma1), to_sparse_coo(scData.sigma2)

        X0, X1, X2 = scData.X0, scData.X1, scData.X2
        L0 = torch.sparse.mm(sigma1, sigma1.t())

        if self.sc_size == 0:
            assert (X0.size()[0] == L0.size()[0])
            return [[X0], [L0.coalesce().indices()], [L0.coalesce().values()]], scData.label

        if self.sc_size == 1:
            L1 = torch.sparse.mm(sigma1.t(), sigma1)
            assert (X0.size()[0] == L0.size()[0])
            assert (X1.size()[0] == L1.size()[0])
            return [[X0, X1], [L0.coalesce().indices(), L1.coalesce().indices()],
                    [L0.coalesce().values(), L1.coalesce().values()]], scData.label

        L1 = torch.sparse.FloatTensor.add(torch.sparse.mm(sigma1.t(), sigma1), torch.sparse.mm(sigma2, sigma2.t()))
        L2 = torch.sparse.mm(sigma2.t(), sigma2)

        # splitting the sparse tensor as pooling cannot return sparse and to make preparation for minibatching easier
        assert (X0.size()[0] == L0.size()[0])
        assert (X1.size()[0] == L1.size()[0])
        assert (X2.size()[0] == L2.size()[0])

        return [[X0, X1, X2],
                [L0.coalesce().indices(), L1.coalesce().indices(), L2.coalesce().indices()],
                [L0.coalesce().values(), L1.coalesce().values(), L2.coalesce().values()]], scData.label


    def collated_data_to_batch(self, samples):
        LapacianData = [self.sc_data_to_lapacian(scData) for scData in samples]
        Lapacians, labels = map(list, zip(*LapacianData))
        labels = torch.cat(labels, dim = 0)
        X_batch, I_batch, V_batch, batch_size = self.sanitize_input_for_batch(Lapacians)

        return (X_batch, I_batch, V_batch, batch_size), labels


    def sanitize_input_for_batch(self, input_list):

        X, L_i, L_v = [*zip(*input_list)]
        X, L_i, L_v = [*zip(*X)], [*zip(*L_i)], [*zip(*L_v)]

        X_batch, I_batch, V_batch, batch_size = self.make_batch(X, L_i, L_v)

        del X, L_v, L_i

        return X_batch, I_batch, V_batch, batch_size


    def make_batch(self, X, L_i, L_v):

        X_batch, I_batch, V_batch, batch_size = [], [], [], []
        for i in range(self.sc_size + 1):
            x, i, v, batch = self.batch(X[i], L_i[i], L_v[i])
            X_batch.append(x)
            I_batch.append(i)
            V_batch.append(v)
            batch_size.append(batch)

        # I_batch and V_batch form the indices and values of coo_sparse tensor but sparse tensors
        # cant be stored so storing them as two separate tensors
        return X_batch, I_batch, V_batch, batch_size


    def batch(self, x_list, L_i_list, l_v_list):
        feature_batch = torch.cat(x_list, dim=0)

        sizes = [*map(lambda x: x.size()[0], x_list)]
        L_i_list = list(L_i_list)
        mx = 0
        for i in range(1, len(sizes)):
            mx += sizes[i - 1]
            L_i_list[i] += mx
        I_cat = torch.cat(L_i_list, dim=1)
        V_cat = torch.cat(l_v_list, dim=0)
        # lapacian_batch = torch.sparse_coo_tensor(L_cat, V_cat)
        batch = [[i for _ in range(sizes[i])] for i in range(len(sizes))]
        batch = torch.tensor([i for sublist in batch for i in sublist], device=DEVICE)
        return feature_batch, I_cat, V_cat, batch
