import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, n_agent):
        super(SelfAttention, self).__init__()
        # TODO (DOUBLE CHECK)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, n_agents, embed_dim = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, n_agents, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, n_agents, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, n_agents, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, n_agents, embed_dim)

        output = self.out_proj(attn_output)

        return output
        
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.encoding[:, :seq_len, :].to(x.device)
        return x

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        # TODO (DOUBLE CHECK)
        self.ln1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.relu(self.ln1(x))
        x = self.ln2(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_agent, hidden_dim=None, max_len=5000):
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim

        super(Encoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_heads, n_agent)
        self.mlp = FeedForward(embed_dim, hidden_dim)

    def forward(self, x):
        x = self.positional_encoding(x)
        attn_output = self.attn(x)
        x = self.layer_norm_1(x, attn_output)
        ff_output = self.mlp(x)
        x = self.layer_norm_2(x, ff_output)
        return x
