# Implementation of classification transformer
# The model will be a combination of multiple transformer blocks
# Input -> word embedding -> position embedding -> transformer blocks -> output sequence -> prediction(just do average)
# transformer layers are permutation independent.
# Position embedding: Add vectors for positions. Drawback: new to see all vectors during training BUT easy to implement
# Position encoding: Choose function that maps positions to real values
import torch
import torch.nn.functional as F
from torch import nn

from transformer import Transformer

device = "cuda" if torch.cuda.is_available() else "cpu"


class CTransformer(nn.Module):
    def __init__(
        self, k, heads, depth, seq_length, num_tokens, num_classes, max_pool=False
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(embedding_dim=k, num_embeddings=num_tokens)
        self.pos_emb = nn.Embedding(embedding_dim=k, num_embeddings=seq_length)
        self.max_pool = max_pool

        blocks = []
        for i in range(depth):
            blocks.append(Transformer(k=k, heads=heads))
        self.tblocks = nn.Sequential(*blocks)

        self.to_probs = nn.Linear(k, num_classes)

    def forward(self, x):
        """
        Forward pass
        """
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        positions = torch.arange(t, device=device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        # Average pool to get class probs
        if self.max_pool:
            x = self.to_probs(x.max(dim=1)[0])
        else:
            x = self.to_probs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
