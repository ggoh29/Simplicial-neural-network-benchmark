import torch

class DatasetBatcher:

    def __init__(self, simplicial_complex_size=2, normalise_fn = None):
        self.sc_size = simplicial_complex_size
        self.normalise_fn = normalise_fn


    def sc_data_to_lapacian(self, scData):

        def to_sparse_coo(matrix):
            indices = matrix[0:2]
            values = matrix[2:3].squeeze()
            return torch.sparse_coo_tensor(indices, values)

        sigma1, sigma2 = to_sparse_coo(scData.sigma1), to_sparse_coo(scData.sigma2)

        X0, X1, X2 = scData.X0, scData.X1, scData.X2
        L0 = torch.sparse.mm(sigma1, sigma1.t())

        if self.normalise_fn is not None:
            L0 = self.normalise_fn(L0)

        if self.sc_size == 0:
            assert (X0.size()[0] == L0.size()[0])
            return [[X0], [L0.coalesce().indices()], [L0.coalesce().values()]], scData.label


        if self.sc_size == 1:
            L1 = torch.sparse.mm(sigma1.t(), sigma1)

            if self.normalise_fn is not None:
                L1 = self.normalise_fn(L1)

            assert (X0.size()[0] == L0.size()[0])
            assert (X1.size()[0] == L1.size()[0])
            return [[X0, X1], [L0.coalesce().indices(), L1.coalesce().indices()],
                    [L0.coalesce().values(), L1.coalesce().values()]], scData.label

        L1 = torch.sparse.FloatTensor.add(torch.sparse.mm(sigma1.t(), sigma1), torch.sparse.mm(sigma2, sigma2.t()))
        L2 = torch.sparse.mm(sigma2.t(), sigma2)

        if self.normalise_fn is not None:
            L1 = self.normalise_fn(L1)
            L2 = self.normalise_fn(L2)

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
        batch = torch.tensor([i for sublist in batch for i in sublist])
        return feature_batch, I_cat, V_cat, batch