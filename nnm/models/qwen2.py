import torch
from torch import nn
from nnm.layers.rope import QwenRoPE
from nnm.layers.norm import Qwen2RMSNorm

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

def expand_kv_heads(x, num_kv_groups):
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, num_kv_groups, seq_len, head_dim)
    return x.reshape(batch, -1, seq_len, head_dim)

class Qwen2Attention(nn.Module):
    def __init__(self, *, embed_dim, num_attn_heads, num_kv_heads, position_encoder):
        super().__init__()
        assert embed_dim % num_attn_heads == 0
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_attn_heads
        self.num_attn_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_attn_heads // self.num_kv_heads
        self.position_encoder = position_encoder
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv_proj = nn.Linear(embed_dim, 2 * num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, cache=False, position=None, attn_mask=None):
        batch, seq_len = x.shape[:-1]

        q = self.q_proj(x)
        k, v = self.kv_proj(x).split(self.num_kv_heads * self.head_dim, dim=-1)
        q = q.reshape(batch, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.position_encoder(q, cache=cache, position=position)
        k = self.position_encoder(k, cache=cache, position=position)

        k = expand_kv_heads(k, self.num_kv_groups)
        v = expand_kv_heads(v, self.num_kv_groups)
        attn_weight = (q @ k.transpose(2, 3) * self.scale).softmax(dim=-1)
        o = attn_weight @ v
        o = self.o_proj(o.transpose(1, 2).reshape(batch, seq_len, self.embed_dim))
        return o

class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self, *, position_encoder,
        embed_dim=896, intermediate_size=4864, num_attn_heads=14, num_kv_heads=2, eps=1e-06,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.position_encoder = position_encoder
        self.attn = Qwen2Attention(
            embed_dim=embed_dim, num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads,
            position_encoder=position_encoder,
        )
        self.mlp = Qwen2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size)
        self.norm_1 = Qwen2RMSNorm(embed_dim=embed_dim, eps=eps)
        self.norm_2 = Qwen2RMSNorm(embed_dim=embed_dim, eps=eps)

    def forward(self, x, attn_mask):
        y = self.norm_1(x)
        y = self.attn(y, attn_mask=attn_mask)
        x = x + y

        y = self.norm_2(x)
        y = self.mlp(y)
        x = x + y

        return x

class Qwen2Model(nn.Module):
    def __init__(self, *, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeds = nn.Embedding(vocab_size, embed_dim)