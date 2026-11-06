import math
from functools import partial

import torch
from torch import nn

from nnm.layers.attention import CrossAttention, SelfAttention
from nnm.layers.embedding import merge_placeholders
from nnm.layers.mlp import MLP
from nnm.layers.transformer import CrossTransformerBlock, TransformerBlock
from nnm.models.davit import DaViT
from nnm.models.pretrained import PretrainedModel


class BartEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_positions, norm_eps, offset=2):
        super().__init__()
        self.offset = offset
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_positions + offset, hidden_size)
        self.layernorm_embedding = nn.LayerNorm(hidden_size, eps=norm_eps)

    def forward(self, input_ids):
        positions = torch.arange(self.offset, self.offset + input_ids.shape[1], device=input_ids.device)
        return self.layernorm_embedding(self.embed_tokens(input_ids) + self.embed_positions(positions))


def build_encoder_layer(hidden_size, num_heads, intermediate_size, norm_eps):
    return TransformerBlock(
        hidden_size,
        SelfAttention(
            hidden_size,
            num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=True,
            o_bias=True,
            causal=False,
        ),
        MLP(hidden_size, intermediate_size, "gelu", bias=True),
        norm_layer=partial(nn.LayerNorm, eps=norm_eps),
    )


def build_decoder_layer(hidden_size, num_heads, intermediate_size, norm_eps):
    return CrossTransformerBlock(
        hidden_size,
        SelfAttention(
            hidden_size,
            num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=True,
            o_bias=True,
            causal=True,
        ),
        CrossAttention(
            hidden_size,
            num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=True,
            o_bias=True,
        ),
        MLP(hidden_size, intermediate_size, "gelu", bias=True),
        norm_layer=partial(nn.LayerNorm, eps=norm_eps),
    )


class Florence2MultiModalProjector(nn.Module):
    def __init__(self, vision_dim, projection_dim, max_positions, max_temporal, norm_eps):
        super().__init__()
        self.image_projection = nn.Linear(vision_dim, projection_dim, bias=False)
        self.image_proj_norm = nn.LayerNorm(projection_dim, eps=norm_eps)
        self.row_embeddings = nn.Embedding(max_positions, vision_dim // 2)
        self.column_embeddings = nn.Embedding(max_positions, vision_dim - vision_dim // 2)
        self.register_buffer("temporal_embedding", self._cosine_embedding(max_temporal, vision_dim))

    @staticmethod
    def _cosine_embedding(max_positions, embed_dim):
        half = embed_dim // 2
        frequencies = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000) / half))
        angles = torch.arange(max_positions, dtype=torch.float32)[:, None] * frequencies[None]
        embedding = torch.empty(max_positions, embed_dim)
        embedding[:, 0::2] = angles.sin()
        embedding[:, 1::2] = angles.cos()
        return embedding

    def _position_embedding(self, height, width, device):
        rows = self.row_embeddings(torch.arange(height, device=device))
        columns = self.column_embeddings(torch.arange(width, device=device))
        rows = rows[:, None].expand(height, width, -1)
        columns = columns[None].expand(height, width, -1)
        return torch.cat([rows, columns], dim=-1).permute(2, 0, 1)[None].float()

    def forward(self, image_features):
        height, width = image_features.shape[-2:]
        positions = self._position_embedding(height, width, image_features.device).to(image_features.dtype)
        tokens = (image_features + positions).flatten(2).transpose(1, 2)
        temporal = self.temporal_embedding[:1].to(tokens.dtype)[None]
        tokens = (tokens + temporal).unsqueeze(1)
        spatial = tokens.mean(dim=2)
        temporal = tokens.mean(dim=1)
        features = torch.cat([spatial, temporal], dim=1)
        return self.image_proj_norm(self.image_projection(features))


class Florence2(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        vision_config = config.get("vision_config")
        text_config = config.get("text_config")
        vision_dim = vision_config.get("embed_dim", vision_config.get("hidden_size", 96))
        if isinstance(vision_dim, (list, tuple)):
            vision_dim = vision_dim[-1]
        projection_dim = vision_config.get("projection_dim", 1024)
        hidden_size = text_config.get("hidden_size", text_config.get("d_model", 768))
        num_heads = text_config.get("num_attention_heads", 12)
        intermediate_size = text_config.get("intermediate_size", text_config.get("encoder_ffn_dim", hidden_size * 4))
        norm_eps = text_config.get("layer_norm_eps", text_config.get("norm_eps", 1e-5))
        vocab_size = text_config.get("vocab_size", 51200)
        max_positions = text_config.get("max_position_embeddings", 1024)
        encoder_layers = text_config.get("encoder_layers", text_config.get("num_hidden_layers", 12))
        decoder_layers = text_config.get("decoder_layers", 12)
        self.image_token_id = config.get("image_token_id", 0)
        self.hidden_size = hidden_size

        self.vision_tower = DaViT(vision_config)
        self.multi_modal_projector = Florence2MultiModalProjector(
            projection_dim,
            hidden_size,
            vision_config.get("max_position_embeddings", 1024),
            vision_config.get("max_temporal_embeddings", 1024),
            norm_eps,
        )
        self.language_model = nn.Module()
        self.language_model.encoder = nn.Module()
        self.language_model.encoder.embed_tokens = BartEmbeddings(vocab_size, hidden_size, max_positions, norm_eps)
        self.language_model.encoder.layers = nn.ModuleList(
            [build_encoder_layer(hidden_size, num_heads, intermediate_size, norm_eps) for _ in range(encoder_layers)]
        )
        self.language_model.encoder.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.language_model.decoder = nn.Module()
        self.language_model.decoder.embed_tokens = BartEmbeddings(vocab_size, hidden_size, max_positions, norm_eps)
        self.language_model.decoder.layers = nn.ModuleList(
            [build_decoder_layer(hidden_size, num_heads, intermediate_size, norm_eps) for _ in range(decoder_layers)]
        )
        self.language_model.decoder.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_image_features(self, pixel_values):
        hidden, _ = self.vision_tower.forward_features(pixel_values)
        return self.multi_modal_projector(hidden)

    def forward(self, pixel_values, input_ids, attention_mask=None, decoder_input_ids=None):
        image_features = self.get_image_features(pixel_values)
        text_embeds = self.language_model.encoder.embed_tokens(input_ids)
        placeholder = (input_ids == self.image_token_id).unsqueeze(-1)
        hidden = merge_placeholders(text_embeds, placeholder, image_features.reshape(-1, self.hidden_size))
        attn_mask = None
        if attention_mask is not None:
            attn_mask = torch.where(attention_mask.bool()[:, None, None, :], 0.0, float("-inf"))
        for layer in self.language_model.encoder.layers:
            hidden = layer(hidden, attn_mask=attn_mask)
        memory = self.language_model.encoder.norm(hidden)

        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        decoder_hidden = self.language_model.decoder.embed_tokens(decoder_input_ids)
        for layer in self.language_model.decoder.layers:
            decoder_hidden = layer(decoder_hidden, memory)
        decoder_hidden = self.language_model.decoder.norm(decoder_hidden)
        return self.lm_head(decoder_hidden)
