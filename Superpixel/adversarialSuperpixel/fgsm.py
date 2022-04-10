import torch

def change_offset_of_edge_and_tri(X, batch, node_offset, shape):
    batch_unique = batch.unique(sorted=True)
    batch_unique_count = torch.stack([(batch == batch_u).sum() for batch_u in batch_unique]).tolist()
    new_offset = []
    acc = 0
    for n_offset, other_offset in zip(node_offset, batch_unique_count):
        new_offset.append(torch.ones((other_offset, shape)) * acc)
        acc += n_offset
    X_offset = torch.cat(new_offset, dim=0).long()
    X += X_offset
    return X


def add_edge_and_tri_offset(feature_tuple):
    simplicialComplex, label = feature_tuple

    batch_0 = simplicialComplex.unpack_batch()[0]
    batch_unique_0 = batch_0.unique(sorted=True)
    batch_unique_0_count = torch.stack([(batch_0 == batch_u).sum() for batch_u in batch_unique_0]).tolist()
    batch_unique_0_offset = batch_unique_0_count

    if simplicialComplex.X1 is not None:
        _, batch_1, batch_2 = simplicialComplex.unpack_batch()
        X1 = simplicialComplex.X1
        simplicialComplex.X1 = change_offset_of_edge_and_tri(X1, batch_1, batch_unique_0_offset, 2)

        X2 = simplicialComplex.X2
        simplicialComplex.X2 = change_offset_of_edge_and_tri(X2, batch_2, batch_unique_0_offset, 3)

    return simplicialComplex, label


def add_edge_and_tri_features(simplicialComplex):
    if simplicialComplex.X1 is not None:
        X0, X1, X2 = simplicialComplex.unpack_features()

        X1_i, X1_j = X0[X1[:, 0]], X0[X1[:, 1]]
        X1 = torch.cat([X1_i, X1_j], dim=1)
        simplicialComplex.X1 = X1

        X2_i, X2_j, X2_k = X0[X2[:, 0]], X0[X2[:, 1]], X0[X2[:, 2]]
        X2 = torch.cat([X2_i, X2_j, X2_k], dim=1)
        simplicialComplex.X2 = X2
    return simplicialComplex


def fgsm_attack(simplicialComplex, epsilon, data_grad, targeted=True):
    pixel_index = torch.tensor([0, 1, 2])
    data_grad = torch.index_select(data_grad.cpu(), 1, pixel_index)
    sign_data_grad = data_grad.sign() * (1 - int(targeted) * 2)
    X0 = simplicialComplex.X0.detach()
    X0_coordinates = torch.index_select(X0, 1, torch.tensor([3, 4]))
    X0_pixels = torch.index_select(X0, 1, pixel_index) + epsilon * sign_data_grad
    X0_pixels = torch.clamp(X0_pixels, 0, 1)
    X0 = torch.cat([X0_pixels, X0_coordinates], dim=1)
    simplicialComplex.X0 = X0.detach()
    return simplicialComplex


def set_grad(simplicialComplex):
    simplicialComplex.X0.requires_grad = True
    if simplicialComplex.X1 is not None:
        simplicialComplex.X1.requires_grad = True
    if simplicialComplex.X2 is not None:
        simplicialComplex.X2.requires_grad = True
    return simplicialComplex


def aggregate_grad(features, X_grad, blank_grad, size):
    slices = [[i for i in range(5)], [i for i in range(5, 10)], [i for i in range(10, 15)]]
    index_lst = []
    for j in range(size):
        index = features[:, j]
        index_lst.append(index)
        slice = torch.index_select(X_grad, 1, torch.tensor(slices[j]))
        blank_grad = blank_grad.index_add_(0, index, slice)
    return blank_grad
    # indexes = torch.cat(index_lst, 0)
    # indexes_u = indexes.unique(sorted=True)
    # indexes = torch.stack([(indexes == index_u).sum() for index_u in indexes_u]).unsqueeze(1)
    # return torch.div(blank_grad, indexes)