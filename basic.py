# Implementation of basic self attention

import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2], [3, 4]])
x = x.unsqueeze(0)  # converting to tensor of dimensions (b,t,k)

raw_weights = torch.bmm(x, x.transpose(1, 2))
softmax_fn = torch.nn.Softmax(dim=2)

weights = softmax_fn(raw_weights.double())

y = torch.bmm(weights, x.double()) # Need to convert to double as softmax does not work for floats

print(y)

# Additional tricks

##
