import unittest
import torch
from skimage.future import graph
from skimage.segmentation import slic
import numpy as np
import networkx as nx
from utils import triangle_to_edge_matrix, edge_to_node_matrix
from constants import DEVICE, TEST_IMAGE_1, TEST_IMAGE_2
from dataset.ImageToDataset import ImageToSimplicialComplex
from dataset.EdgeFlow import EdgeFlowSC, MakeSC

class MyTestCase(unittest.TestCase):
    def test_sparse_mm_yields_same_result_as_dense_mm(self):
        image = TEST_IMAGE_1

        image = np.array(image)
        superpixel = slic(image, n_segments=100, compactness=0.75, start_label=1)
        rag = graph.rag_mean_color(image, superpixel)

        sigma1 = edge_to_node_matrix(rag.edges(), rag.nodes)

        L0 = torch.matmul(sigma1, sigma1.T)
        sigma1_coo = sigma1.to_sparse()
        L0_coo = torch.sparse.mm(sigma1_coo, sigma1_coo.t())
        L0_coo = L0_coo.to_dense()
        self.assertTrue(torch.all(torch.eq(L0, L0_coo)).item())


    def test_sparse_add_yields_same_result_as_dense_add(self):
        image = TEST_IMAGE_2

        image = np.array(image)
        superpixel = slic(image, n_segments=150, compactness=0.75, start_label=1)
        rag = graph.rag_mean_color(image, superpixel)
        triangles = [*filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(rag))]

        sigma1 = edge_to_node_matrix(rag.edges(), rag.nodes)
        sigma2 = triangle_to_edge_matrix(triangles, rag.edges)

        sigma1_coo, sigma2_coo = sigma1.to_sparse(), sigma2.to_sparse()

        L1= torch.matmul(sigma1.T, sigma1) + torch.matmul(sigma2, sigma2.T)
        L1_coo = torch.sparse.FloatTensor.add(torch.sparse.mm(sigma1_coo.t(), sigma1_coo),
                                                torch.sparse.mm(sigma2_coo, sigma2_coo.t()))

        L1_coo = L1_coo.to_dense()
        self.assertTrue(torch.all(torch.eq(L1, L1_coo)).item())


    def test_Lapacian_0_generated_correctly(self):
        image = TEST_IMAGE_2

        image = np.array(image)
        superpixel = slic(image, n_segments=150, compactness=0.75, start_label=1)
        rag = graph.rag_mean_color(image, superpixel)

        # Ensuring this function works correctly
        sigma1 = edge_to_node_matrix(rag.edges(), rag.nodes)

        L0 = torch.matmul(sigma1, sigma1.T)

        D = [[0 for _ in range(len(rag.nodes))] for _ in range(len(rag.nodes))]
        D = torch.tensor(D, dtype=torch.float, device=DEVICE)

        for x_node in rag.nodes:
            D[x_node-1][x_node-1] += len(rag.adj[x_node])

        G = nx.Graph()
        G.add_edges_from(rag.edges)
        A = nx.adjacency_matrix(G, nodelist=range(1, len(rag.nodes)+1)).todense()

        L0_test = D - A

        self.assertTrue(torch.all(torch.eq(L0, L0_test)).item())


    def test_Lapacian_1_generated_correctly(self):
        image = TEST_IMAGE_2

        image = np.array(image)
        superpixel = slic(image, n_segments=150, compactness=0.75, start_label=1)
        rag = graph.rag_mean_color(image, superpixel)
        triangles = [*filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(rag))]
        sigma1 = edge_to_node_matrix(rag.edges(), rag.nodes)

        # Ensuring this function works correctly
        sigma2 = triangle_to_edge_matrix(triangles, rag.edges)

        L1 = torch.matmul(sigma1.T, sigma1) + torch.matmul(sigma2, sigma2.T)

        I_1 = torch.eye(len(rag.edges)) * 2

        A_lower = [[0 for _ in range(len(rag.edges))] for _ in range(len(rag.edges))]
        A_lower = torch.tensor(A_lower, dtype=torch.float, device=DEVICE)

        edge_l = [e for e in rag.edges]
        edges = {edge_l[i]: i for i in range(len(edge_l))}

        for i in range(0, len(edges)):
            for j in range(i + 1, len(edges)):
                x_1, y_1 = edge_l[i]
                x_2, y_2 = edge_l[j]
                if x_1 == x_2 or y_1 == y_2:
                    A_lower[edges[(x_1, y_1)]][edges[(x_2, y_2)]] += 1
                    A_lower[edges[(x_2, y_2)]][edges[(x_1, y_1)]] += 1
                elif x_1 == y_2 or y_1 == x_2:
                    A_lower[edges[(x_1, y_1)]][edges[(x_2, y_2)]] -= 1
                    A_lower[edges[(x_2, y_2)]][edges[(x_1, y_1)]] -= 1

        D = [[0 for _ in range(len(rag.edges))] for _ in range(len(rag.edges))]
        D = torch.tensor(D, dtype=torch.float, device=DEVICE)

        A_upper = [[0 for _ in range(len(rag.edges))] for _ in range(len(rag.edges))]
        A_upper = torch.tensor(A_upper, dtype=torch.float, device=DEVICE)

        for i, j, k in triangles:
            D[edges[(i, j)]][edges[(i, j)]] += 1
            D[edges[(j, k)]][edges[(j, k)]] += 1
            D[edges[(i, k)]][edges[(i, k)]] += 1

            face_1 = ((i, j), (k, i))
            face_2 = ((k, i), (j, k))
            face_3 = ((i, j), (j, k))
            for face1, face2 in [face_1, face_2, face_3]:
                if face1 in edges and face2 in edges:
                    A_upper[edges[face1]][edges[face2]] -= 1
                    A_upper[edges[face2]][edges[face1]] -= 1
                elif face1 in edges and face2[::-1] in edges:
                    A_upper[edges[face1]][edges[face2[::-1]]] += 1
                    A_upper[edges[face2[::-1]]][edges[face1]] += 1
                elif face2 in edges and face1[::-1] in edges:
                    A_upper[edges[face2]][edges[face1[::-1]]] += 1
                    A_upper[edges[face1[::-1]]][edges[face2]] += 1
                else:
                    A_upper[edges[face2[::-1]]][edges[face1[::-1]]] += 1
                    A_upper[edges[face1[::-1]]][edges[face2[::-1]]] += 1

        L1_test = D - A_upper + I_1 + A_lower

        self.assertTrue(torch.all(torch.eq(L1, L1_test)).item())


    def test_edge_flow_Lapacian_0_generated_correctly(self):
        I2SC = ImageToSimplicialComplex(100, EdgeFlowSC, 0)
        image = TEST_IMAGE_2
        image = np.array(image)
        nodes, edges, triangles, node_features = I2SC.image_to_features(image)
        _, L_i, L_v = I2SC.features_to_lapacians(nodes, edges, triangles, node_features)
        L0 = torch.sparse_coo_tensor(L_i[0], L_v[0])
        L0 = L0.to_dense()

        D = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
        D = torch.tensor(D, dtype=torch.float, device=DEVICE)

        adj = {}

        for x, y in edges:
            if x not in adj:
                adj[x] = []
            adj[x].append(y)
            if y not in adj:
                adj[y] = []
            adj[y].append(x)


        for x_node in nodes:
            D[x_node-1][x_node-1] += len(adj[x_node])

        G = nx.Graph()
        G.add_edges_from(edges)
        A = nx.adjacency_matrix(G, nodelist=range(1, len(nodes)+1)).todense()

        L0_test = D - A

        self.assertTrue(torch.all(torch.eq(L0, L0_test)).item())


    def test_edge_flow_Lapacian_1_generated_correctly(self):
        I2SC = ImageToSimplicialComplex(100, EdgeFlowSC, 2)
        image = TEST_IMAGE_2
        image = np.array(image)
        nodes, edges, triangles, node_features = I2SC.image_to_features(image)
        _, L_i, L_v = I2SC.features_to_lapacians(nodes, edges, triangles, node_features)
        L1 = torch.sparse_coo_tensor(L_i[1], L_v[1])
        L1 = L1.to_dense()

        I_1 = torch.eye(len(edges)) * 2

        A_lower = [[0 for _ in range(len(edges))] for _ in range(len(edges))]
        A_lower = torch.tensor(A_lower, dtype=torch.float, device=DEVICE)

        edge_l = [e for e in edges]
        edges = {edge_l[i]: i for i in range(len(edge_l))}

        for i in range(0, len(edges)):
            for j in range(i + 1, len(edges)):
                x_1, y_1 = edge_l[i]
                x_2, y_2 = edge_l[j]
                if x_1 == x_2 or y_1 == y_2:
                    A_lower[edges[(x_1, y_1)]][edges[(x_2, y_2)]] += 1
                    A_lower[edges[(x_2, y_2)]][edges[(x_1, y_1)]] += 1
                elif x_1 == y_2 or y_1 == x_2:
                    A_lower[edges[(x_1, y_1)]][edges[(x_2, y_2)]] -= 1
                    A_lower[edges[(x_2, y_2)]][edges[(x_1, y_1)]] -= 1

        D = [[0 for _ in range(len(edges))] for _ in range(len(edges))]
        D = torch.tensor(D, dtype=torch.float, device=DEVICE)

        A_upper = [[0 for _ in range(len(edges))] for _ in range(len(edges))]
        A_upper = torch.tensor(A_upper, dtype=torch.float, device=DEVICE)

        for i, j, k in triangles:

            bl = [1, 1, 1]

            if (i,j) in edges:
                e1 = (i,j)
            else:
                e1 = (j,i)
                bl[0] = -1

            if (j,k) in edges:
                e2 = (j,k)
            else:
                e2 = (k,j)
                bl[1] = -1

            if (k,i) in edges:
                e3 = (k,i)
            else:
                e3 = (i,k)
                bl[2] = -1

            D[edges[e1]][edges[e1]] += 1
            D[edges[e2]][edges[e2]] += 1
            D[edges[e3]][edges[e3]] += 1

            face_1 = (e1, e3, bl[0] * bl[2])
            face_2 = (e3, e2, bl[1] * bl[2])
            face_3 = (e1, e2, bl[0] * bl[1])
            for i in range(3):
                face1, face2, bl = [face_1, face_2, face_3][i]
                if face1 in edges and face2 in edges:
                    A_upper[edges[face1]][edges[face2]] -= 1 * bl
                    A_upper[edges[face2]][edges[face1]] -= 1 * bl
                elif face1 in edges and face2[::-1] in edges:
                    A_upper[edges[face1]][edges[face2[::-1]]] += 1 * bl
                    A_upper[edges[face2[::-1]]][edges[face1]] += 1 * bl
                elif face2 in edges and face1[::-1] in edges:
                    A_upper[edges[face2]][edges[face1[::-1]]] += 1 * bl
                    A_upper[edges[face1[::-1]]][edges[face2]] += 1 * bl
                else:
                    A_upper[edges[face2[::-1]]][edges[face1[::-1]]] += 1 * bl
                    A_upper[edges[face1[::-1]]][edges[face2[::-1]]] += 1 * bl

        L1_test = D - A_upper + I_1 + A_lower

        self.assertTrue(torch.all(torch.eq(L1, L1_test)).item())



if __name__ == '__main__':
    unittest.main()
