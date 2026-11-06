import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.attention import InterleavedRotarySelfAttention
from nnm.layers.dit import LearnableTimestepEmbedder
from nnm.layers.mlp import GatedMLP
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import LearnableRoPE2D
from nnm.layers.transformer import TransformerBlock
from nnm.layers.vit import patchify, unpatchify
from nnm.models.pretrained import PretrainedModel


class JiTTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.get("dim")
        hidden_dim = config.get("hidden_dim", dim * 4)
        num_heads = config.get("num_heads")
        num_layers = config.get("num_layers")
        qk_norm = config.get("qk_norm", False)
        num_temb_freqs = config.get("num_temb_freqs", dim // 2)
        num_coord_dims = config.get("num_coord_dims", 2)
        bottleneck_dim = config.get("bottleneck_dim")
        num_classes = config.get("num_classes", 0)
        self.x_embedder = nn.Linear(bottleneck_dim, dim, bias=False)
        self.timestep_embedder = LearnableTimestepEmbedder(dim, num_temb_freqs) if num_temb_freqs else None
        self.class_embedding = nn.Embedding(num_classes + 1, dim) if num_classes else None
        self.rope = LearnableRoPE2D(num_layers, num_heads, dim // num_heads, num_coord_dims)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    InterleavedRotarySelfAttention(
                        dim,
                        num_heads,
                        qkv_bias=False,
                        o_bias=False,
                        qk_norm=qk_norm,
                        causal=False,
                        norm_layer=RMSNorm,
                    ),
                    GatedMLP(dim, hidden_dim),
                    norm_layer=RMSNorm,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, bottleneck_dim, bias=False)

    def forward(self, hidden, timesteps, class_labels, height, width):
        hidden = self.x_embedder(hidden).flatten(1, 2)
        prefix = []
        if timesteps is not None:
            prefix.append(self.timestep_embedder(timesteps)[:, None])
        if class_labels is not None:
            prefix.append(self.class_embedding(class_labels)[:, None])
        pad = 2 if len(prefix) != 1 else 1
        if prefix:
            hidden = torch.cat([torch.zeros_like(prefix[0]) for _ in range(pad)] + prefix + [hidden], dim=1)
        else:
            hidden = torch.cat([torch.zeros_like(hidden[:, :1]) for _ in range(pad)] + [hidden], dim=1)
        cos, sin = self.rope(height, width)
        num_prefix = pad + len(prefix)
        padded_cos = F.pad(cos, (0, 0, 0, 0, num_prefix, 0), value=1.0)
        padded_sin = F.pad(sin, (0, 0, 0, 0, num_prefix, 0), value=0.0)
        for layer, layer_cos, layer_sin in zip(self.layers, padded_cos, padded_sin):
            rotary_embeddings = (layer_cos.permute(1, 0, 2)[None], layer_sin.permute(1, 0, 2)[None])
            hidden = layer(hidden, rotary_embeddings=rotary_embeddings)
        hidden = self.norm(hidden)[:, num_prefix:]
        return self.head(hidden)


class JiT(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        patch_size = config.get("patch_size", 16)
        data_dim = config.get("data_dim", config.get("in_channels", 3))
        bottleneck_dim = config.get("bottleneck_dim", config.get("dim"))
        self.patch_size = patch_size
        self.data_dim = data_dim
        patch_dim = patch_size * patch_size * data_dim
        self.bottleneck_down = nn.Linear(patch_dim, bottleneck_dim, bias=False)
        self.bottleneck_up = nn.Linear(bottleneck_dim, patch_dim, bias=False)
        self.transformer = JiTTransformer(config)

    def forward(self, pixel_values, timesteps=None, class_labels=None):
        batch, _, height, width = pixel_values.shape
        patch_height, patch_width = height // self.patch_size, width // self.patch_size
        hidden = patchify(pixel_values, self.patch_size)
        hidden = self.bottleneck_down(hidden.reshape(batch, patch_height, patch_width, -1))
        hidden = self.transformer(hidden, timesteps, class_labels, patch_height, patch_width)
        hidden = self.bottleneck_up(hidden)
        hidden = hidden.reshape(batch, -1, self.patch_size * self.patch_size * self.data_dim)
        return torch.tanh(unpatchify(hidden, (height, width), self.patch_size))
