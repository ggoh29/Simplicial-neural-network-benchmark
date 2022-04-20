import unittest

import torch
import numpy as np
from nn_utils import convert_to_CoChain, torch_sparse_to_scipy_sparse, scipy_sparse_to_torch_sparse, to_sparse_coo
from scipy import sparse
from models import SuperpixelSCConv, test_SAN, test_SAT, test_SCConv, test_SCN, flow_SCConv, flow_SAN, flow_SAT, flow_SCN
from CoChain import CoChain


def Bunch_github_processing(B1, B2):
    # This is directly from Eric Bunch's code https://github.com/AmFamMLTeam/simplicial-2-complex-cnns/blob/main/mnist-classification/scripts/prepare_data.py#L349
    L0 = B1 @ B1.T
    B1_sum = np.abs(B1).sum(axis=1)
    D0 = sparse.diags(B1_sum.A.reshape(-1), 0)
    B1_sum_inv = 1. / B1_sum
    B1_sum_inv[np.isinf(B1_sum_inv) | np.isneginf(B1_sum_inv)] = 0
    D0_inv = sparse.diags(B1_sum_inv.A.reshape(-1), 0)
    L0 = D0_inv @ L0
    L0factor = (-1) * sparse.diags((1 / (B1_sum_inv + 1)).A.reshape(-1), 0)
    L0bias = sparse.identity(n=D0.shape[0])
    L0 = L0factor @ L0 + L0bias
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
    B2_sum_inv = np.squeeze(np.asarray(B2_sum_inv))
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


def generate_oriented_flow_pair():
    # This is the complex from slide 19 of https://crisbodnar.github.io/files/mml_talk.pdf
    B1 = torch.tensor([
        [-1, -1, 0, 0, 0, 0],
        [+1, 0, -1, 0, 0, +1],
        [0, +1, 0, -1, 0, -1],
        [0, 0, +1, +1, -1, 0],
        [0, 0, 0, 0, +1, 0],
    ], dtype=torch.float)

    B2 = torch.tensor([
        [-1, 0],
        [+1, 0],
        [0, +1],
        [0, -1],
        [0, 0],
        [+1, +1],
    ], dtype=torch.float)

    X1 = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [-1.0]], dtype=torch.float)
    T2 = torch.diag(torch.tensor([+1.0, +1.0, +1.0, +1.0, -1.0, -1.0], dtype=torch.float))

    X0, X2 = torch.zeros((B1.shape[0], 1)), torch.zeros((B2.shape[1], 1))
    label = torch.tensor([0])

    cochain1 = CoChain(X0, X1, X2, B1.to_sparse(), B2.to_sparse(), label)

    B1 = (B1 @ T2)
    B2 = (T2 @ B2)
    X1 = T2 @ X1

    cochain2 = CoChain(X0, X1, X2, B1.to_sparse(), B2.to_sparse(), label)

    return cochain1, cochain2, T2


def process_cochain(cochain, processor):
    g = processor.process(cochain)
    feature_dct, _ = processor.batch([g])
    feature_dct = processor.clean_features(feature_dct)
    feature_dct = processor.repair(feature_dct)
    return feature_dct


class MyTestCase(unittest.TestCase):

    def test_Bunch_processor_sparse_returns_same_result_as_original(self):
        nb_nodes = 200
        adj_i = torch.triu_indices(nb_nodes, nb_nodes, 1)
        adj_v = torch.tensor(np.random.binomial(1, 0.10, size=(adj_i.shape[1])), dtype=torch.float)
        adj_matrix = torch.sparse_coo_tensor(adj_i, adj_v, (nb_nodes, nb_nodes)).to_dense().to_sparse().cuda()

        features = torch.tensor([[1, 1] for _ in range(nb_nodes)]).cuda()
        labels = torch.tensor([0 for _ in range(nb_nodes)]).cuda()
        sc_data = convert_to_CoChain(adj_matrix, features, labels)

        X0, X1, X2 = sc_data.X0, sc_data.X1, sc_data.X2
        x0, x1, x2 = X0.shape[0], X1.shape[0], X2.shape[0]

        b1, b2 = to_sparse(sc_data.b1, (x0, x1)), to_sparse(sc_data.b2, (x1, x2))

        B1, B2 = torch_sparse_to_scipy_sparse(b1), torch_sparse_to_scipy_sparse(b2)
        results = Bunch_github_processing(B1, B2)
        processor = SuperpixelSCConv[0]
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

        self.assertTrue(torch.allclose(L0, L0_b, atol=1e-5))
        self.assertTrue(torch.allclose(L1, L1_b, atol=1e-5))
        self.assertTrue(torch.allclose(L2, L2_b, atol=1e-5))
        self.assertTrue(torch.allclose(B2D3, B2D3_b, atol=1e-5))
        self.assertTrue(torch.allclose(D2B1TD1inv, D2B1TD1inv_b, atol=1e-5))
        self.assertTrue(torch.allclose(D1invB1, D1invB1_b, atol=1e-5))
        self.assertTrue(torch.allclose(B2TD2inv, B2TD2inv_b, atol=1e-5))

    def test_orientation_equivariance_Ebli(self):
        f = torch.nn.Tanh()
        input_size, output_size = 1, 1

        module = test_SCN
        processor_type = module[0]
        model = module[1](input_size, input_size, input_size, output_size, f=f)

        chain1, chain2, T2 = generate_oriented_flow_pair()
        f1 = process_cochain(chain1, processor_type)
        f2 = process_cochain(chain2, processor_type)

        _, result1, _ = model(f1)
        _, result2, _ = model(f2)
        self.assertTrue(torch.allclose(T2 @ result1, result2, atol=1e-5))

    def test_orientation_equivariance_Bunch(self):
        f = torch.nn.Tanh()
        input_size, output_size = 1, 1

        module = test_SCConv
        processor_type = module[0]
        model = module[1](input_size, input_size, input_size, output_size, f=f)

        chain1, chain2, T2 = generate_oriented_flow_pair()
        f1 = process_cochain(chain1, processor_type)
        f2 = process_cochain(chain2, processor_type)

        _, result1, _ = model(f1)
        _, result2, _ = model(f2)

        self.assertTrue(torch.allclose(T2 @ result1, result2, atol=1e-5))

    def test_orientation_equivariance_SAT(self):
        f = torch.nn.Tanh()
        input_size, output_size = 1, 1

        module = test_SAT
        processor_type = module[0]
        model = module[1](input_size, input_size, input_size, output_size, f=f)

        chain1, chain2, T2 = generate_oriented_flow_pair()
        f1 = process_cochain(chain1, processor_type)
        f2 = process_cochain(chain2, processor_type)

        _, result1, _ = model(f1)
        _, result2, _ = model(f2)

        self.assertTrue(torch.allclose(T2 @ result1, result2, atol=1e-5))

    def test_orientation_equivariance_SAN(self):
        f = torch.nn.Tanh()
        input_size, output_size = 1, 1

        module = test_SAN
        processor_type = module[0]
        model = module[1](input_size, input_size, input_size, output_size, f=f)

        chain1, chain2, T2 = generate_oriented_flow_pair()
        f1 = process_cochain(chain1, processor_type)
        f2 = process_cochain(chain2, processor_type)

        _, result1, _ = model(f1)
        _, result2, _ = model(f2)
        print(result2, result1)
        self.assertTrue(torch.allclose(T2 @ result1, result2, atol=1e-5))

    def test_orientation_invariant_Ebli(self):
        f = torch.nn.Tanh()
        input_size, output_size = 1, 2

        module = flow_SCN
        processor_type = module[0]
        model = module[1](input_size, input_size, input_size, output_size, f=f)

        chain1, chain2, T2 = generate_oriented_flow_pair()
        f1 = process_cochain(chain1, processor_type)
        f2 = process_cochain(chain2, processor_type)

        result1 = model(f1)
        result2 = model(f2)

        self.assertTrue(torch.allclose(result1, result2, atol=1e-5))


    def test_orientation_invariant_Bunch(self):
        f = torch.nn.Tanh()
        input_size, output_size = 1, 2

        module = flow_SCConv
        processor_type = module[0]
        model = module[1](input_size, input_size, input_size, output_size, f=f)

        chain1, chain2, T2 = generate_oriented_flow_pair()
        f1 = process_cochain(chain1, processor_type)
        f2 = process_cochain(chain2, processor_type)

        result1 = model(f1)
        result2 = model(f2)

        self.assertTrue(torch.allclose(result1, result2, atol=1e-5))

    def test_orientation_invariant_SAT(self):
        f = torch.nn.Tanh()
        input_size, output_size = 1, 2

        module = flow_SAT
        processor_type = module[0]
        model = module[1](input_size, input_size, input_size, output_size, f=f)

        chain1, chain2, T2 = generate_oriented_flow_pair()
        f1 = process_cochain(chain1, processor_type)
        f2 = process_cochain(chain2, processor_type)

        result1 = model(f1)
        result2 = model(f2)

        self.assertTrue(torch.allclose(result1, result2, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
