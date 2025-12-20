import math
import torch
import torch.nn as nn

def spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # A: sparse_coo_tensor (N,N), X: (N,F)
    return torch.sparse.mm(A, X)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.uniform_(-stdv, stdv)

    def forward(self, adj_sp: torch.Tensor, x: torch.Tensor):
        x = x.float()
        support = x @ self.weight
        out = spmm(adj_sp, support)
        if self.bias is not None:
            out = out + self.bias
        return out

class GCNStack(nn.Module):
    """
    3-layer GCN like many baseline codes.
    Return (h1,h2,h3)
    """
    def __init__(self, dim):
        super().__init__()
        self.g1 = GraphConvolution(dim, dim)
        self.g2 = GraphConvolution(dim, dim)
        self.g3 = GraphConvolution(dim, dim)

    def forward(self, adj, feat):
        h1 = torch.relu(self.g1(adj, feat))
        h2 = torch.relu(self.g2(adj, h1))
        h3 = torch.relu(self.g3(adj, h2))
        return h1, h2, h3
