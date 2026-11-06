import torch
from functools import partial
from torch import nn
from torch.nn import functional as F

from nnm.layers.attention import SelfAttention
from nnm.layers.mlp import MLP
from nnm.layers.transformer import TransformerBlock, VisionBlock
from nnm.layers.vit import AttentionPool, PatchEmbed2D
from nnm.models.pretrained import PretrainedModel


class SigLIP2VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.get("image_size", 224)
        patch_size = config.get("patch_size", 16)
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_positions = config.get(
            "num_positions", config.get("num_patches", config.get("max_num_patches", (image_size // patch_size) ** 2))
        )
        self.patch_size = patch_size
        self.num_positions = num_positions
        self.patch_embedding = PatchEmbed2D(
            kernel_size=patch_size,
            in_channels=config.get("num_channels", 3),
            embed_dim=hidden_size,
            bias=True,
            norm=False,
        )
        self.position_embedding = nn.Embedding(num_positions, hidden_size)

    def forward(self, pixel_values):
        hidden, (height, width) = self.patch_embedding(pixel_values)
        position = self.position_embedding.weight[None]
        if height * width != self.num_positions:
            grid = int(self.num_positions**0.5)
            position = position.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
            position = F.interpolate(position, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
            position = position.permute(0, 2, 3, 1).reshape(1, height * width, -1)
        return hidden + position[:, : height * width]


class SigLIP2TextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        vocab_size = config.get("vocab_size", 32000)
        max_positions = config.get("max_position_embeddings", 64)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)


class SigLIP2TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        norm_eps = config.get("layer_norm_eps", config.get("norm_eps", 1e-6))
        activation = config.get("hidden_act", "gelu_pytorch_tanh")
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
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
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden, attn_mask=None):
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=attn_mask)
        return hidden


class SigLIP2VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        norm_eps = config.get("layer_norm_eps", config.get("norm_eps", 1e-6))
        activation = config.get("hidden_act", "gelu_pytorch_tanh")
        self.embeddings = SigLIP2VisionEmbeddings(config)
        self.encoder = nn.ModuleList(
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
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                )
                for _ in range(num_layers)
            ]
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.use_head = config.get("vision_use_head", True)
        self.head = AttentionPool(hidden_size, num_heads, intermediate_size, activation, norm_eps) if self.use_head else None

    def forward(self, pixel_values):
        hidden = self.embeddings(pixel_values)
        for layer in self.encoder:
            hidden = layer(hidden)
        hidden = self.post_layernorm(hidden)
        pooled = self.head(hidden) if self.head is not None else hidden[:, 0]
        return {"last_hidden_state": hidden, "pooler_output": pooled}


class SigLIP2TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        self.embeddings = SigLIP2TextEmbeddings(config)
        self.encoder = SigLIP2TextEncoder(config)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=config.get("layer_norm_eps", config.get("norm_eps", 1e-6)))
        self.head = nn.Linear(hidden_size, config.get("projection_size", hidden_size), bias=True)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embeddings(input_ids)
        mask = None
        if attention_mask is not None:
            mask = torch.where(attention_mask.bool(), 0.0, float("-inf"))[:, None, None, :]
        hidden = self.encoder(hidden, mask)
        hidden = self.final_layer_norm(hidden)
        pooled = self.head(hidden[:, -1])
        return {"last_hidden_state": hidden, "pooler_output": pooled}


class SigLIP2(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.vision_model = SigLIP2VisionModel(config)
        self.text_model = SigLIP2TextModel(config)
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.logit_bias = nn.Parameter(torch.zeros(1))

    def get_image_features(self, pixel_values):
        return self.vision_model(pixel_values)["pooler_output"]

    def get_text_features(self, input_ids, attention_mask=None):
        return self.text_model(input_ids, attention_mask)["pooler_output"]

    def forward(self, pixel_values, input_ids, attention_mask=None):
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t() + self.logit_bias
        return {"logits_per_image": logits_per_image, "logits_per_text": logits_per_image.t()}
