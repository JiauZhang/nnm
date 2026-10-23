import torch
import torch.nn.functional as F
from torch import nn


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x, cos, sin):
    return x * cos[:, None] + rotate_half(x) * sin[:, None]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        freqs = position_ids[:, :, None].float() * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def forward(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.num_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask):
        batch, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(2))
        query, key = apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=self.scale)
        return self.Wo(out.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        hidden, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(hidden) * gate)


class Layer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.attn_norm = nn.Identity() if index == 0 else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_types = list(config.layer_types)
        self.windows = {
            layer_type: config.local_attention // 2 if layer_type == "sliding_attention" else None
            for layer_type in set(self.layer_types)
        }
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([Layer(config, index) for index in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.rope = nn.ModuleDict(
            {
                layer_type: RotaryEmbedding(
                    config.hidden_size // config.num_attention_heads, config.rope_parameters[layer_type]["rope_theta"]
                )
                for layer_type in self.windows
            }
        )

    def build_mask(self, attention_mask, length, window, device):
        keep = torch.ones(1, 1, 1, length, dtype=torch.bool, device=device)
        if attention_mask is not None:
            keep = attention_mask.to(device).bool()[:, None, None, :]
        if window is None:
            return keep
        positions = torch.arange(length, device=device)
        return keep & ((positions[:, None] - positions[None, :]).abs() <= window)[None, None]

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden = self.embeddings(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]

        length = input_ids.shape[1]
        ropes = {layer_type: self.rope[layer_type](position_ids, hidden.dtype) for layer_type in self.windows}
        masks = {
            layer_type: self.build_mask(attention_mask, length, window, input_ids.device)
            for layer_type, window in self.windows.items()
        }
        for layer, layer_type in zip(self.layers, self.layer_types):
            cos, sin = ropes[layer_type]
            hidden = layer(hidden, cos, sin, masks[layer_type])
        return self.final_norm(hidden)
