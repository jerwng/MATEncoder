import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, n_agent):
        super(SelfAttention, self).__init__()
        # TODO

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        # TODO

class Encoder(nn.module):
    def __init__(self, embed_dim, n_heads, n_agent, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = 1 * embed_dim
        
        super(Encoder, self).__init__()
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_heads, n_agent)
        self.mlp = FeedForward(embed_dim, hidden_dim)


