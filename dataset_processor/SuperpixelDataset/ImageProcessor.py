import torch
from skimage.segmentation import slic
from utils import triangle_to_edge_matrix, edge_to_node_matrix
from utils import sparse_to_tensor, tensor_to_sparse
from skimage import color
import numpy as np
from constants import DEVICE

def stl(t):
    "Shape to list"
    return list(t.shape)

class SCData:

    def __init__(self, X0, X1, X2, b1, b2, label):
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2
        # b1 and b2 can either be sparse or dense but since python doesn't really have overloading, doing this instead
        if b1.is_sparse:
            b1 = sparse_to_tensor(b1)
        self.b1 = b1

        if b2.is_sparse:
            b2 = sparse_to_tensor(b2)
        self.b2 = b2

        self.label = label


    def __str__(self):
        name = f"SCData(X0={stl(self.X0)}, X1={stl(self.X1)}, X2={stl(self.X2)}," \
               f" b1={stl(self.b1)}, b2={stl(self.b2)}, label={stl(self.label)})"
        return name

    def __repr__(self):
        name = f"SCData(X0={stl(self.X0)}, X1={stl(self.X1)}, X2={stl(self.X2)}," \
               f" b1={stl(self.b1)}, b2={stl(self.b2)}, label={stl(self.label)})"
        return name

    def __eq__(self, other):
        x0 = torch.all(torch.eq(self.X0, other.X0)).item()
        x1 = torch.all(torch.eq(self.X1, other.X1)).item()
        x2 = torch.all(torch.eq(self.X2, other.X2)).item()
        s1 = torch.all(torch.eq(self.b1, other.b1)).item()
        s2 = torch.all(torch.eq(self.b2, other.b2)).item()
        l0 = torch.all(torch.eq(self.label, other.label)).item()
        return all([x0, x1, x2, s1, s2, l0])


class ProcessImage:

    def __init__(self, superpixel_size, edgeflow):
        self.spixel_size = superpixel_size
        self.edgeflow = edgeflow


    def image_to_features(self, image):
        # Saving to disk must be cpu rather than cuda
        image, label = image
        label = torch.tensor([label], device=DEVICE)
        image = (image.double().numpy())

        is_rgb = False
        if image.shape[0] == 3:
            is_rgb = True
            # Convert from (3, N, N) matrix to (N, N, 3)
            image_rgb = np.stack(image, axis=-1)
            # Has rgb pixel values, so grayscale it. Otherwise superpixeling will give errors
            image = color.rgb2gray(image_rgb)
        elif image.shape[0] == 1:
            image = image[0]

        superpixel = slic(image, n_segments=self.spixel_size, compactness=1, start_label=1)

        if is_rgb:
            image = image_rgb

        nodes, edges, triangles, node_features = self.edgeflow.convert_graph(image, superpixel)

        X0 = torch.tensor(node_features, dtype=torch.float, device=DEVICE)

        X1 = []
        for x, y in edges:
            X1.append(node_features[x - 1] + node_features[y - 1])

        X1 = torch.tensor(X1, dtype=torch.float, device=DEVICE)

        X2 = []
        for i, j, k in triangles:
            X2.append(node_features[i - 1] + node_features[j - 1] + node_features[k - 1])

        X2 = torch.tensor(X2, dtype=torch.float, device=DEVICE)

        b1 = edge_to_node_matrix(edges, nodes)
        b1 = b1.to_sparse()

        b2 = triangle_to_edge_matrix(triangles, edges)
        b2 = b2.to_sparse()

        return SCData(X0, X1, X2, b1, b2, label)



