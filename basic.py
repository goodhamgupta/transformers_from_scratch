# Implementation of basic self attention

import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2], [3, 4]])
x = x.unsqueeze(0)  # converting to tensor of dimensions (b,t,k)

#raw_weights = torch.bmm(x, x.transpose(1, 2))
#softmax_fn = torch.nn.Softmax(dim=2)
#
#weights = softmax_fn(raw_weights.double())
#
#y = torch.bmm(weights, x.double()) # Need to convert to double as softmax does not work for floats
#
#print(y)

# Additional tricks

## Queries, Keys and Values

# - x_i compared to every other vector to get own weights y_i
# - x_i compared to every other vector to get j-th vector weights y_j
# - x_i used as part of weighted sum to compute each output vector once weights have been established

# Three different matrices. W_k, W_q and W_v for each of the above operations
# q = W_q x_i  k = W_k x_i v = W_v x_i

# Operations
## raw_weights = q^T . k
## weights = softmax(raw_weights)
## y = sum_{i=1}^{n} weights . W_v

# Scale dot product by sqrt(k) where k is the dimension of the input tensor
# Multi-head attention => Use multiple Q, K and V matrices. Typically based on dim of input tensor. If dim is 256 and you use 8 matrices, each matrix will have dimension 32x32.

W_q = torch.rand(2,2).unsqueeze(0)
W_k = torch.rand(2,2).unsqueeze(0)
W = torch.rand(2,2).unsqueeze(0)
q = torch.bmm(W_q.double(), x.double())
print(q)
