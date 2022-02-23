import torch
from skimage.segmentation import slic
from utils import triangle_to_edge_matrix, edge_to_node_matrix
from models.SCData import SCData
from skimage import color
import numpy as np


class ImageProcessor:

    def __init__(self, superpixel_size, edgeflow):
        self.spixel_size = superpixel_size
        self.edgeflow = edgeflow

    def image_to_features(self, image):
        nodes, edges, triangles, node_features, label = self.image_to_complex(image)
        features = self.complex_to_features(nodes, edges, triangles, node_features, label)
        return features

    def image_to_complex(self, image):
        # Saving to disk must be cpu rather than cuda
        image, label = image
        label = torch.tensor([label])
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
        return nodes, edges, triangles, node_features, label

    def complex_to_features(self, nodes, edges, triangles, node_features, label):
        X0 = torch.tensor(node_features, dtype=torch.float)
        X1 = torch.tensor(edges, dtype=torch.long) - 1
        X2 = torch.tensor(triangles, dtype=torch.long) - 1

        X1_i, X1_j = X0[X1[:, 0]], X0[X1[:, 1]]
        X1 = torch.cat([X1_i, X1_j], dim=1)

        X2_i, X2_j, X2_k = X0[X2[:, 0]], X0[X2[:, 1]], X0[X2[:, 2]]
        X2 = torch.cat([X2_i, X2_j, X2_k], dim=1)

        b1 = edge_to_node_matrix(edges, nodes)
        b1 = b1.to_sparse()

        b2 = triangle_to_edge_matrix(triangles, edges)
        b2 = b2.to_sparse()

        return SCData(X0, X1, X2, b1, b2, label)


class AdversarialImageProcessor(ImageProcessor):

    def complex_to_features(self, nodes, edges, triangles, node_features, label):
        X0 = torch.tensor(node_features, dtype=torch.float)
        X1 = torch.tensor(edges, dtype=torch.long) - 1
        X2 = torch.tensor(triangles, dtype=torch.long) - 1

        b1 = edge_to_node_matrix(edges, nodes)
        b1 = b1.to_sparse()

        b2 = triangle_to_edge_matrix(triangles, edges)
        b2 = b2.to_sparse()

        return SCData(X0, X1, X2, b1, b2, label)


class OrientatedImageProcessor(ImageProcessor):

    def complex_to_features(self, nodes, edges, triangles, node_features, label):
        X0 = torch.tensor(node_features, dtype=torch.float)
        X1 = torch.tensor(edges, dtype=torch.long) - 1
        X2 = torch.tensor(triangles, dtype=torch.long) - 1

        X1_i, X1_j = X0[X1[:, 0]], X0[X1[:, 1]]
        X1 = X1_i - X1_j
        X1_pixels = torch.index_select(X1, 1, torch.tensor([0, 1, 2]))
        X1_dist = torch.index_select(X1, 1, torch.tensor([3, 4]))
        X1_dist_i, X1_dist_j = X1_dist[:, 0], X1_dist[:, 1]
        X1_dist = torch.pow(torch.pow(X1_dist_i, 2) + torch.pow(X1_dist_j, 2), 0.5)
        X1 = torch.div(X1_pixels, X1_dist.unsqueeze(1))

        X0 = torch.zeros((X0.shape[0], 3))

        X2 = torch.zeros(X2.shape)

        b1 = edge_to_node_matrix(edges, nodes)
        b1 = b1.to_sparse()

        b2 = triangle_to_edge_matrix(triangles, edges)
        b2 = b2.to_sparse()

        return SCData(X0, X1, X2, b1, b2, label)
