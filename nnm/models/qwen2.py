import torch
from torch import nn
from nnm.layers.positional import QwenRoPE

class Qwen2MLP(nn.Module):
    def __init__(self, *, embed_dim, intermediate_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen2Attention(nn.Module):
    def __init__(self, *, embed_dim, num_attn_heads, num_kv_heads, position_encoder):
        assert embed_dim % num_attn_heads == 0
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_attn_heads
        self.num_attn_heads = num_attn_heads
        self.position_encoder = position_encoder
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv_proj = nn.Linear(embed_dim, 2 * num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, cache=False, position=None):
        q = self.q_proj(x) * self.scale
        k, v = self.kv_proj(x).split(2, dim=-1)
        q = self.position_encoder(q, cache=cache, position=position)
        k = self.position_encoder(k, cache=cache, position=position)
        o = (q @ k.transpose(-1, -2)).softmax(dim=-1) @ v
        o = self.o_proj(o)
        return o

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, *, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

class Qwen2Model(nn.Module):
    def __init__(self, *, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeds = nn.Embedding(vocab_size, embed_dim)