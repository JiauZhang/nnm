import torch
from functools import partial
from torch import nn
from torch.nn import functional as F

from nnm.layers.attention import SelfAttention
from nnm.layers.mlp import MLP
from nnm.layers.transformer import VisionBlock
from nnm.layers.vit import PatchEmbed2D
from nnm.models.pretrained import PretrainedModel


class ViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        image_size = config.get("image_size", 224)
        patch_size = config.get("patch_size", 16)
        num_channels = config.get("num_channels", 3)
        num_patches = (image_size // patch_size) ** 2
        self.patch_embeddings = PatchEmbed2D(
            kernel_size=patch_size,
            in_channels=num_channels,
            embed_dim=hidden_size,
            bias=True,
            norm=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))

    def forward(self, pixel_values):
        hidden, (height, width) = self.patch_embeddings(pixel_values)
        hidden = torch.cat([self.cls_token.expand(hidden.shape[0], -1, -1), hidden], dim=1)
        if hidden.shape[1] != self.position_embeddings.shape[1]:
            hidden = hidden + self._interpolate_positions(height, width)
        else:
            hidden = hidden + self.position_embeddings
        return hidden

    def _interpolate_positions(self, height, width):
        grid_height = grid_width = int((self.position_embeddings.shape[1] - 1) ** 0.5)
        cls_position = self.position_embeddings[:, :1]
        patch_positions = self.position_embeddings[:, 1:].reshape(1, grid_height, grid_width, -1).permute(0, 3, 1, 2)
        patch_positions = F.interpolate(patch_positions, size=(height, width), mode="bicubic", align_corners=False)
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(1, height * width, -1)
        return torch.cat([cls_position, patch_positions], dim=1)


class ViT(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_layers = config.get("num_hidden_layers", config.get("depth", 12))
        num_heads = config.get("num_attention_heads", config.get("num_heads", 12))
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        layer_norm_eps = config.get("layer_norm_eps", 1e-12)
        activation = config.get("hidden_act", config.get("activation", "gelu"))
        num_labels = config.get("num_labels", 0)

        self.embeddings = ViTEmbeddings(config)
        self.layers = nn.ModuleList(
            [
                VisionBlock(
                    hidden_size,
                    SelfAttention(
                        hidden_size,
                        num_heads,
                        head_dim=hidden_size // num_heads,
                        qkv_bias=True,
                        o_bias=True,
                        causal=False,
                    ),
                    MLP(hidden_size, intermediate_size, activation, bias=True),
                    norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.classifier = nn.Linear(hidden_size, num_labels) if num_labels else None

    def forward(self, pixel_values):
        hidden = self.embeddings(pixel_values)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)

    def classify(self, pixel_values):
        hidden = self.forward(pixel_values)
        if self.classifier is None:
            return hidden[:, 0]
        return self.classifier(hidden[:, 0])
