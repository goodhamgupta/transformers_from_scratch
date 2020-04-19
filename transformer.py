# Implementation of transformer
# Transformer

# Applies following operations in sequence:
# -> Input
# -> Self attention
# -> Layer Norm
# -> MLP
# -> Layer norm
# -> Output
# Need to combine attention with MLP. Apply layer norm and residuals

from .complete_self_attention import SelfAttentionWide, SelfAttentionNarrow


class Transformer(nn.Module):
    """
    Transformer module
    """

    def __init__(self, k, heads=8):
        self.attention = SelfAttentionWide(k, heads)
        self.layer_1 = nn.LayerNorm(k)
        self.layer_2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.Relu(),
            nn.Linear(4*k, k)
        )

    def forward(self, x):
        att_output = self.attention(x)
        norm = self.layer_1(att_output)

        ff_output = self.ff(norm)
        return self.layer_2(ff_output + x) # Add x as the residual component
