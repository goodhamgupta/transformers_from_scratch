# Implementation of Generation transformer(autoregressive)


dot = torch.bmm(queries, keys.transpose(1,2))

indices = torch.triu_indices(t, t, offset=1)
dot[:, indices[0], indices[1]] = float("-inf")

dot = F.softmax(dot, dim=2)
