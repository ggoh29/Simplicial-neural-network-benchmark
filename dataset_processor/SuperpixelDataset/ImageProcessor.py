import torch
from skimage.segmentation import slic
from utils import triangle_to_edge_matrix, edge_to_node_matrix
from dataset_processor.SCData import SCData
from skimage import color
import numpy as np
from constants import DEVICE


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



