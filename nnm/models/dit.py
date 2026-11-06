import torch
from torch import nn

from nnm.layers.dit import DiTBlock, FinalLayer, TimestepEmbedder
from nnm.layers.vit import PatchEmbed2D
from nnm.models.pretrained import PretrainedModel


class DiT(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        patch_size = config.patch_size
        in_channels = config.get("in_channels", 3)
        learn_sigma = config.get("learn_sigma", True)
        num_classes = config.get("num_classes", 1000)

        self.x_embedder = PatchEmbed2D(
            kernel_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True, norm=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes, hidden_size)
        num_patches = (config.image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, config.num_attention_heads, config.get("intermediate_size", hidden_size * 4))
                for _ in range(config.num_hidden_layers)
            ]
        )
        out_channels = in_channels * 2 if learn_sigma else in_channels
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.out_channels = out_channels
        self.patch_size = patch_size

    def forward(self, hidden_states, timestep, class_labels):
        hidden, (height, width) = self.x_embedder(hidden_states)
        hidden = hidden + self.pos_embed[:, : hidden.shape[1]]
        condition = self.t_embedder(timestep) + self.y_embedder(class_labels)
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = self.final_layer(hidden, condition)
        batch, _, _ = hidden.shape
        hidden = hidden.reshape(batch, height, width, self.patch_size, self.patch_size, self.out_channels)
        hidden = hidden.permute(0, 5, 1, 3, 2, 4)
        return hidden.reshape(batch, self.out_channels, height * self.patch_size, width * self.patch_size)
