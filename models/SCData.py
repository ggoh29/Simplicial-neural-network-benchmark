import torch
from utils import sparse_to_tensor, tensor_to_sparse
from models.nn_utils import to_sparse_coo

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
        b_5 = to_sparse_coo(b1)

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