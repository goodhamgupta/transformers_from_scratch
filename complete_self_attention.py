import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionWide(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        # Convert the output from the above operations back to dimension of size k
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, input_x):
        """
        Function to compute the self-attention
        """
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(input_x).view(b, t, h, k)
        keys = self.tokeys(input_x).view(b, t, h, k)
        values = self.tovalues(input_x).view(b, t, h, k)

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # Scaling dot product for more efficient memory usage

        queries = queries / (k ** 0.25)
        keys = keys / (k ** 0.25)

        dot_prod = torch.bmm(queries, keys.transpose(1, 2))
        softmax = F.softmax(dot_prod, dim=2)

        out = torch.bmm(softmax, values).view(b, h, t, k)

        # unify output from multiple heads

        final_out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(final_out)


class SelfAttentionNarrow(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        dim = k // heads
        self.toqueries = nn.Linear(dim, dim, bias=False)
        self.tokeys = nn.Linear(dim, dim, bias=False)
        self.tovalues = nn.Linear(dim, dim, bias=False)

        # Convert the output from the above operations back to dimension of size k
        self.unifyheads = nn.Linear(heads * dim, k)

    def forward(self, input_x):
        """
        Function to compute the self-attention
        """
        b, t, k = x.size()
        h = self.heads
        dim = k // h
        x = x.view(b, t, h, dim)

        queries = self.toqueries(input_x)
        keys = self.tokeys(input_x)
        values = self.tovalues(input_x)

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, dim)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, dim)
        values = values.transpose(1, 2).contiguous().view(b * h, t, dim)

        # Scaling dot product for more efficient memory usage

        queries = queries / (k ** 0.25)
        keys = keys / (k ** 0.25)

        dot_prod = torch.bmm(queries, keys.transpose(1, 2))
        softmax = F.softmax(dot_prod, dim=2)

        out = torch.bmm(softmax, values).view(b, h, t, dim)

        # unify output from multiple heads

        final_out = out.transpose(1, 2).contiguous().view(b, t, h * s)
        return self.unifyheads(final_out)
