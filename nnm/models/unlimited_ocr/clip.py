import math

import torch
from torch import nn
from torch.nn import functional as F


from nnm.layers.transformer import MLP
from nnm.layers.vit import resize_abs_pos


def get_abs_pos(abs_pos, target_tokens):
    dim = abs_pos.shape[-1]
    cls_token, old_pos_embed = abs_pos.squeeze(0)[:1], abs_pos.squeeze(0)[1:]
    source_size = int(math.sqrt(abs_pos.squeeze(0).shape[0] - 1))
    target_size = int(math.sqrt(target_tokens))
    if source_size == target_size:
        return abs_pos
    old_pos_embed = old_pos_embed.view(1, source_size, source_size, dim)
    new_pos_embed = resize_abs_pos(old_pos_embed, target_size).view(target_size * target_size, dim)
    return torch.cat([cls_token, new_pos_embed], dim=0).view(1, target_size * target_size + 1, dim)


class ClipEmbeddings(nn.Module):
    def __init__(self, *, width, image_size, patch_size, in_chans=3):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.patch_embedding = nn.Conv2d(in_chans, width, kernel_size=patch_size, stride=patch_size, bias=False)
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches + 1, width)
        self.register_buffer("position_ids", torch.arange(num_patches + 1).expand((1, -1)))

    def forward(self, patch_embeds):
        batch = patch_embeds.shape[0]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        return embeddings + get_abs_pos(self.position_embedding(self.position_ids), embeddings.shape[1])


class ClipAttention(nn.Module):
    def __init__(self, *, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out_proj(out)


class ClipBlock(nn.Module):
    def __init__(self, *, dim, num_heads, hidden_dim, norm_eps):
        super().__init__()
        self.self_attn = ClipAttention(dim=dim, num_heads=num_heads)
        self.mlp = MLP(dim, hidden_dim, "quick_gelu", bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class ClipTransformer(nn.Module):
    def __init__(self, *, width, depth, num_heads, hidden_dim, norm_eps):
        super().__init__()
        self.layers = nn.ModuleList(
            [ClipBlock(dim=width, num_heads=num_heads, hidden_dim=hidden_dim, norm_eps=norm_eps) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ClipEncoder(nn.Module):
    def __init__(self, *, width, image_size, patch_size, depth, num_heads, hidden_dim, pre_layernorm_eps, norm_eps):
        super().__init__()
        self.embeddings = ClipEmbeddings(width=width, image_size=image_size, patch_size=patch_size)
        self.pre_layrnorm = nn.LayerNorm(width, eps=pre_layernorm_eps)
        self.transformer = ClipTransformer(
            width=width, depth=depth, num_heads=num_heads, hidden_dim=hidden_dim, norm_eps=norm_eps
        )

    def forward(self, x, patch_embeds):
        if patch_embeds is None:
            patch_embeds = self.embeddings.patch_embedding(x)
        return self.transformer(self.pre_layrnorm(self.embeddings(patch_embeds)))
