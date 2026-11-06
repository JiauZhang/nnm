import torch
from torch import nn

from nnm.layers.conv import ResBlock2D
from nnm.layers.dit import DiTBlock
from nnm.layers.norm import LayerNorm2d
from nnm.layers.vq import VectorQuantizer
from nnm.models.pretrained import PretrainedModel


class VarVQVAE(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim", 128))
        in_channels = config.get("in_channels", 3)
        num_res_blocks = config.get("num_res_blocks", 2)
        downsample = config.get("downsample", [2, 2])
        norm_eps = config.get("norm_eps", 1e-6)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, 3, padding=1),
            *[ResBlock2D(hidden_size, norm_eps) for _ in range(num_res_blocks)],
            *[nn.Conv2d(hidden_size, hidden_size, 2, stride=step) for step in downsample],
        )
        self.decoder = nn.Sequential(
            *[nn.ConvTranspose2d(hidden_size, hidden_size, step, stride=step) for step in downsample],
            *[ResBlock2D(hidden_size, norm_eps) for _ in range(num_res_blocks)],
            LayerNorm2d(hidden_size, eps=norm_eps),
            nn.Conv2d(hidden_size, in_channels, 3, padding=1),
        )
        self.quantizer = VectorQuantizer(
            hidden_size,
            config.get("codebook_size", 4096),
            config.get("codebook_dim"),
        )

    def forward(self, pixel_values):
        latent = self.encoder(pixel_values)
        codes = self.quantizer.encode(latent.reshape(latent.shape[0], latent.shape[1], -1))
        quantized = self.quantizer.decode(codes).reshape(latent.shape)
        return self.decoder(quantized), codes


class VAR(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        vocab_size = config.get("vocab_size", config.get("codebook_size", 4096))
        num_classes = config.get("num_classes", 1000)
        num_scales = config.get("num_scales", 10)
        max_position = config.get("max_position_embeddings", 1024)
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        self.class_embed = nn.Embedding(num_classes, hidden_size)
        self.scale_embed = nn.Embedding(num_scales, hidden_size)
        self.word_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_position, hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, token_ids, class_labels, scale, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.word_embed(token_ids) + self.pos_embed(position_ids)
        condition = self.class_embed(class_labels) + self.scale_embed(scale)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.head(hidden)
