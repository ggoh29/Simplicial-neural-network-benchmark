import unittest

import torch
import numpy as np
from nn_utils import convert_to_SC, torch_sparse_to_scipy_sparse, scipy_sparse_to_torch_sparse, to_sparse_coo
from scipy import sparse
from models import superpixel_Bunch_nn

def Bunch_github_processing(B1, B2):

    L0 = B1 @ B1.T
    B1_sum = np.abs(B1).sum(axis=1)
    D0 = sparse.diags(B1_sum.A.reshape(-1), 0)
    B1_sum_inv = 1. / B1_sum
    B1_sum_inv[np.isinf(B1_sum_inv) | np.isneginf(B1_sum_inv)] = 0
    D0_inv = sparse.diags(B1_sum_inv.A.reshape(-1), 0)
    L0 = L0 @ D0_inv
    L0factor = (-1) * sparse.diags((1 / (B1_sum_inv + 1)).A.reshape(-1), 0)
    L0bias = sparse.identity(n=D0.shape[0])
    L0 = L0 @ L0factor + L0bias
    D1_inv = sparse.diags((B1_sum_inv * 0.5).A.reshape(-1), 0)
    D2diag = np.max(
        (
            abs(B2).sum(axis=1).A.reshape(-1),
            np.ones(shape=(B2.shape[0]))
        ),
        axis=0
    )
    D2 = sparse.diags(D2diag, 0)
    D2inv = sparse.diags(1 / D2diag, 0)
    D3 = (1 / 3.) * sparse.identity(n=B2.shape[1])

    A_1u = D2 - B2 @ D3 @ B2.T
    A_1d = D2inv - B1.T @ D1_inv @ B1
    A_1u_norm = (
                        A_1u
                        +
                        sparse.identity(n=A_1u.shape[0])
                ) @ sparse.diags(1 / (D2.diagonal() + 1), 0)
    A_1d_norm = (
                        D2
                        +
                        sparse.identity(n=D2.shape[0])
                ) @ (A_1d + sparse.identity(n=A_1d.shape[0]))
    # not really L1, but easy to drop in; normalized adjacency
    L1 = A_1u_norm + A_1d_norm

    B2_sum = abs(B2).sum(axis=1)
    B2_sum_inv = 1 / (B2_sum + 1)
    B2_sum_inv = np.squeeze(np.asarray(B2_sum_inv ))
    D5inv = sparse.diags(B2_sum_inv, 0)

    A_2d = sparse.identity(n=B2.shape[1]) + B2.T @ D5inv @ B2
    A_2d_norm = (2 * sparse.identity(n=B2.shape[1])) @ (
            A_2d
            +
            sparse.identity(n=A_2d.shape[0])
    )
    L2 = A_2d_norm  # normalized adjacency

    B2D3 = B2 @ D3
    D2B1TD1inv = (1 / np.sqrt(2.)) * D2 @ B1.T @ D1_inv
    D1invB1 = (1 / np.sqrt(2.)) * D1_inv @ B1
    B2TD2inv = B2.T @ D5inv
    return [L0, L1, L2, B2D3, D2B1TD1inv, D1invB1, B2TD2inv]


def to_sparse(matrix, size):
    indices = matrix[0:2]
    values = matrix[2:3].squeeze()
    return torch.sparse_coo_tensor(indices, values, size)

class MyTestCase(unittest.TestCase):

    def test_Bunch_processor_sparse_returns_same_result_as_original(self):
        nb_nodes = 200
        adj_i = torch.triu_indices(nb_nodes, nb_nodes, 1)
        adj_v = torch.tensor(np.random.binomial(1, 0.10, size=(adj_i.shape[1])), dtype=torch.float)
        adj_matrix = torch.sparse_coo_tensor(adj_i, adj_v, (nb_nodes, nb_nodes)).to_dense().to_sparse().cuda()

        features = torch.tensor([[1, 1] for _ in range(nb_nodes)]).cuda()
        labels = torch.tensor([0 for _ in range(nb_nodes)]).cuda()
        sc_data = convert_to_SC(adj_matrix, features, labels)

        X0, X1, X2 = sc_data.X0, sc_data.X1, sc_data.X2
        x0, x1, x2 = X0.shape[0], X1.shape[0], X2.shape[0]

        b1, b2 = to_sparse(sc_data.b1, (x0, x1)), to_sparse(sc_data.b2, (x1, x2))

        B1, B2 =  torch_sparse_to_scipy_sparse(b1), torch_sparse_to_scipy_sparse(b2)
        results = Bunch_github_processing(B1, B2)
        processor = superpixel_Bunch_nn[0]
        bunch_results = [scipy_sparse_to_torch_sparse(sparse.coo_matrix(matrix)).to_dense() for matrix in results]
        L0_b, L1_b, L2_b, B2D3_b, D2B1TD1inv_b, D1invB1_b, B2TD2inv_b = bunch_results

        sc_object = processor.process(sc_data)

        L0 = to_sparse_coo(sc_object.L0).to_dense()
        L1 = to_sparse_coo(sc_object.L1).to_dense()
        L2 = to_sparse_coo(sc_object.L2).to_dense()
        B2D3 = to_sparse_coo(sc_object.B2D3).to_dense()
        D2B1TD1inv = to_sparse_coo(sc_object.D2B1TD1inv).to_dense()
        D1invB1 = to_sparse_coo(sc_object.D1invB1).to_dense()
        B2TD2inv = to_sparse_coo(sc_object.B2TD2inv).to_dense()
        print(L0)
        print(L0_b)
        self.assertTrue(torch.allclose(L0, L0_b, atol=1e-5))
        self.assertTrue(torch.allclose(L1, L1_b, atol=1e-5))
        self.assertTrue(torch.allclose(L2, L2_b, atol=1e-5))
        self.assertTrue(torch.allclose(B2D3, B2D3_b, atol=1e-5))
        self.assertTrue(torch.allclose(D2B1TD1inv, D2B1TD1inv_b, atol=1e-5))
        self.assertTrue(torch.allclose(D1invB1, D1invB1_b, atol=1e-5))
        self.assertTrue(torch.allclose(B2TD2inv, B2TD2inv_b, atol=1e-5))


    # def test_lapacian_normalisation_function_is_correct(self):
    #
    #     def get_D(L):
    #         L = L.to_dense()
    #         L[L != 0] = 1
    #         adj = torch.triu(L, 1)
    #         adj = adj + adj.T
    #         L = adj.to_sparse()
    #         D = torch.sparse.sum(L, dim=1).to_dense()
    #         return D
    #
    #     def get_D_inv(L):
    #         L = L.to_dense()
    #         L[L != 0] = 1
    #         adj = torch.triu(L, 1)
    #         adj = adj + adj.T
    #         L = adj.to_sparse()
    #         D = torch.sparse.sum(L, dim=1).to_dense()
    #         D = 1 / torch.sqrt(D)
    #         i = [i for i in range(D.shape[0])]
    #         D = torch.sparse_coo_tensor(torch.tensor([i, i]), D)
    #         return D
    #
    #     def normalise_adjacency_u(L, D):
    #         L = L.to_dense()
    #         L[L != 0] = 1
    #         adj = torch.triu(L, 1)
    #         adj = adj + adj.T
    #         L = adj.to_sparse()
    #         D = torch.sparse.sum(L, dim=1).to_dense()
    #         D = 1 / torch.sqrt(D)
    #         i = [i for i in range(D.shape[0])]
    #         D = torch.sparse_coo_tensor(torch.tensor([i, i]), D)
    #         return torch.eye(L.shape[0]) - torch.sparse.mm(torch.sparse.mm(D, L), D).to_dense()
    #
    #     def normalise_adjacency_d(L, D):
    #         L = L.to_dense()
    #         L[L != 0] = 1
    #         adj = torch.triu(L, 1)
    #         adj = adj + adj.T
    #         L = adj.to_sparse()
    #         return torch.sparse.mm(torch.sparse.mm(D, L), D).to_dense()
    #
    #     def deg_k(k, D):
    #         return (k + 1) * torch.sparse.mm(D, D)
    #
    #     nb_nodes = 200
    #     adj_i = torch.triu_indices(nb_nodes, nb_nodes, 1)
    #     adj_v = torch.tensor(np.random.binomial(1, 0.10, size=(adj_i.shape[1])), dtype=torch.float)
    #     adj_matrix = torch.sparse_coo_tensor(adj_i, adj_v, (nb_nodes, nb_nodes)).to_dense().to_sparse().cuda()
    #
    #     features = torch.tensor([[1, 1] for _ in range(nb_nodes)])
    #     labels = torch.tensor([0 for _ in range(nb_nodes)])
    #     sc_data = convert_to_SC(adj_matrix, features, labels)
    #
    #     processor = superpixel_Ebli_nn[0]
    #     sc_object = processor.process(sc_data)
    #
    #     X0, X1, X2 = sc_data.X0, sc_data.X1, sc_data.X2
    #     x0, x1, x2 = X0.shape[0], X1.shape[0], X2.shape[0]
    #
    #     b1, b2 = to_sparse(sc_data.b1, (x0, x1)), to_sparse(sc_data.b2, (x1, x2))
    #
    #     L0 = torch.sparse.mm(b1, b1.t()).to('cpu')
    #     L1_d = torch.sparse.mm(b1.t(), b1).to('cpu')
    #     L1_u = torch.sparse.mm(b2, b2.t()).to('cpu')
    #     L2 = torch.sparse.mm(b2.t(), b2).to('cpu')
    #
    #     D0 = get_D_inv(L0)
    #     L0 = normalise_adjacency_u(L0, D0)
    #
    #     D2_i = get_D_inv(L2)
    #     L2_k = deg_k(2, D2_i).to_dense()
    #     L2 = normalise_adjacency_d(L2, D2_i) + L2_k + torch.eye(L2.shape[0])
    #
    #     L0_ebli = to_sparse_coo(sc_object.L0).to_dense()
    #     L2_ebli = to_sparse_coo(sc_object.L2).to_dense()
    #
    #     print(L0)
    #     print(L0_ebli)
    #
    #     print(L2)
    #     print(L2_ebli)
    #
    #     self.assertTrue(torch.allclose(L0, L0_ebli, atol = 1e-5))






if __name__ == '__main__':
    unittest.main()
