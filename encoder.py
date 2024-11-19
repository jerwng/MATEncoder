import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def init_(module, activate=False):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))
        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.proj(y)
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(1).unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, n_agents, n_embd = x.size()
        pos_encoding = self.encoding[:, :, :seq_len, :].to(x.device)
        pos_encoding = pos_encoding.expand(batch_size, seq_len, n_agents, n_embd)
        return x + pos_encoding

class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class Encoder(nn.Module):
    def __init__(self, state_dim, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, encode_state, action_type='Discrete'):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.action_type = action_type
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.positional_encoding = PositionalEncoding(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))
        self.act_head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))
        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, state, obs):
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings
        x = self.positional_encoding(x)
        rep = self.blocks(x)
        v_loc = self.head(rep)
        logit = self.act_head(rep)
        return v_loc, rep, logit
