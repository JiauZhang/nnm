import torch
from functools import partial
from torch import nn

from nnm.layers.attention import SelfAttention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.mlp import MLP
from nnm.layers.transformer import TransformerBlock, VisionBlock
from nnm.layers.vit import resize_abs_pos
from nnm.models.pretrained import PretrainedModel


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.get("image_size", 224)
        patch_size = config.get("patch_size", 16)
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_channels = config.get("num_channels", 3)
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.class_embedding = nn.Parameter(torch.randn(hidden_size))
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)

    def forward(self, pixel_values=None, patch_embeds=None):
        if patch_embeds is None:
            if pixel_values is None:
                raise ValueError("Either pixel_values or patch_embeds must be provided")
            patch_embeds = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))

        batch = patch_embeds.shape[0]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        position = self.position_embedding.weight[None]
        num_patches = embeddings.shape[1] - 1
        if num_patches != self.num_patches:
            grid = int(self.num_patches**0.5)
            target = int(num_patches**0.5)
            class_position = position[:, :1]
            patch_position = position[:, 1:].reshape(1, grid, grid, -1)
            patch_position = resize_abs_pos(patch_position, target).reshape(1, target * target, -1)
            position = torch.cat([class_position, patch_position], dim=1)
        return embeddings + position


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        vocab_size = config.get("vocab_size", 49408)
        max_positions = config.get("max_position_embeddings", 77)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)


class CLIPEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden, attn_mask=None):
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=attn_mask)
        return hidden


class CLIPVisionModel(nn.Module):
    def __init__(self, config, post_layernorm=True):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        norm_eps = config.get("layer_norm_eps", config.get("norm_eps", 1e-6))
        activation = config.get("hidden_act", "quick_gelu")
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.encoder = CLIPEncoder(
            [
                VisionBlock(
                    hidden_size,
                    SelfAttention(hidden_size, num_heads, qkv_bias=True, o_bias=True, causal=False),
                    MLP(hidden_size, intermediate_size, activation, bias=True),
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                )
                for _ in range(num_layers)
            ]
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=norm_eps) if post_layernorm else None

    def forward(self, pixel_values=None, patch_embeds=None):
        hidden = self.embeddings(pixel_values, patch_embeds)
        hidden = self.pre_layrnorm(hidden)
        hidden = self.encoder(hidden)
        pooled = hidden[:, 0]
        if self.post_layernorm is not None:
            pooled = self.post_layernorm(pooled)
        return {"last_hidden_state": hidden, "pooler_output": pooled}


class CLIPTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        norm_eps = config.get("layer_norm_eps", config.get("norm_eps", 1e-6))
        activation = config.get("hidden_act", "quick_gelu")
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(
            [
                TransformerBlock(
                    hidden_size,
                    SelfAttention(hidden_size, num_heads, qkv_bias=True, o_bias=True, causal=True),
                    MLP(hidden_size, intermediate_size, activation, bias=True),
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embeddings(input_ids)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = make_causal_mask(
                input_ids.shape[1],
                input_ids.shape[1],
                0,
                input_ids.device,
                hidden.dtype,
                attn_mask=attention_mask,
            )
        hidden = self.encoder(hidden, attn_mask)
        hidden = self.final_layer_norm(hidden)
        return {"last_hidden_state": hidden, "pooler_output": hidden[:, -1]}


class CLIP(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        vision_config = config.get("vision_config") or config
        text_config = config.get("text_config") or config
        vision_hidden_size = vision_config.get("hidden_size", vision_config.get("embed_dim"))
        text_hidden_size = text_config.get("hidden_size", text_config.get("embed_dim"))
        projection_dim = config.get("projection_dim", text_hidden_size)
        logit_scale_init_value = config.get("logit_scale_init_value", 2.6592)
        self.vision_model = CLIPVisionModel(vision_config)
        self.text_model = CLIPTextModel(text_config)
        self.visual_projection = nn.Linear(vision_hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init_value)))

    def get_image_features(self, pixel_values=None, patch_embeds=None):
        pooled = self.vision_model(pixel_values=pixel_values, patch_embeds=patch_embeds)["pooler_output"]
        return self.visual_projection(pooled)

    def get_text_features(self, input_ids, attention_mask=None):
        pooled = self.text_model(input_ids, attention_mask)["pooler_output"]
        return self.text_projection(pooled)

    def forward(self, pixel_values, input_ids, attention_mask=None):
        image_embeds = self.get_image_features(pixel_values)
        text_embeds = self.get_text_features(input_ids, attention_mask)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logits_per_text = text_embeds @ image_embeds.t() * self.logit_scale.exp()
        return {
            "logits_per_image": logits_per_text.t(),
            "logits_per_text": logits_per_text,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }
