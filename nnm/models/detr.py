import math
from functools import partial

import torch
from torch import nn

from nnm.layers.attention import CrossAttention, SelfAttention
from nnm.layers.mlp import MLP
from nnm.layers.transformer import TransformerBlock


class DetrSinePositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_boxes(self, boxes):
        dim_t = self.temperature ** (
            2
            * torch.div(torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device), 2, rounding_mode="floor")
            / self.num_pos_feats
        )
        embeddings = []
        for index in range(4):
            angle = boxes[..., index, None] * self.scale / dim_t
            embeddings.append(torch.stack((angle[..., 0::2].sin(), angle[..., 1::2].cos()), dim=-1).flatten(-2))
        y, x, w, h = embeddings
        return torch.cat((y, x, w, h), dim=-1)

    def forward(self, shape, device, dtype, mask=None):
        height, width = shape[-2:]
        if mask is None:
            mask = torch.zeros((1, height, width), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        eps = 1e-6
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = self.temperature ** (
            2
            * torch.div(torch.arange(self.num_pos_feats, dtype=torch.int64, device=device), 2, rounding_mode="floor")
            / self.num_pos_feats
        )
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class DetrEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.0, norm_eps=1e-5):
        super().__init__()
        self.block = TransformerBlock(
            hidden_size,
            SelfAttention(hidden_size, num_heads, dropout=dropout),
            MLP(hidden_size, intermediate_size, "relu", dropout=dropout),
            norm_layer=partial(nn.LayerNorm, eps=norm_eps),
        )

    def forward(self, hidden, attn_mask=None):
        return self.block(hidden, attn_mask=attn_mask)


class DetrEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, intermediate_size, dropout=0.0, norm_eps=1e-5):
        super().__init__()
        self.layers = nn.ModuleList(
            DetrEncoderLayer(hidden_size, num_heads, intermediate_size, dropout, norm_eps) for _ in range(num_layers)
        )

    def forward(self, hidden, pos=None, attn_mask=None):
        for layer in self.layers:
            if pos is not None:
                hidden = hidden + pos
            hidden = layer(hidden, attn_mask)
        return hidden


class DetrDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.0, norm_eps=1e-5):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.cross_attn = CrossAttention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, "relu", dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, memory, query_pos=None, memory_pos=None, self_attn_mask=None, cross_attn_mask=None):
        q = hidden + query_pos if query_pos is not None else hidden
        hidden = self.norm1(hidden + self.dropout(self.self_attn(q, q, q, attn_mask=self_attn_mask)))
        q = hidden + query_pos if query_pos is not None else hidden
        k = memory + memory_pos if memory_pos is not None else memory
        hidden = self.norm2(hidden + self.dropout(self.cross_attn(q, k, k, attn_mask=cross_attn_mask)))
        hidden = self.norm3(hidden + self.dropout(self.mlp(hidden)))
        return hidden


class DetrDecoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_layers,
        intermediate_size,
        num_queries=100,
        dropout=0.0,
        norm_eps=1e-5,
        return_intermediate=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.return_intermediate = return_intermediate
        self.query_embed = nn.Embedding(num_queries, hidden_size)
        self.layers = nn.ModuleList(
            DetrDecoderLayer(hidden_size, num_heads, intermediate_size, dropout, norm_eps) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, 4),
        )

    def forward(self, memory, memory_pos=None, attn_mask=None, cross_attn_mask=None):
        batch = memory.shape[0]
        query = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        reference = inverse_sigmoid(self.bbox_embed(self.norm(query)).sigmoid())

        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                memory,
                query_pos=self.query_embed.weight,
                memory_pos=memory_pos,
                self_attn_mask=attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(query))

        output = self.norm(query)
        if self.return_intermediate:
            return torch.stack(intermediate), reference
        return output, reference
