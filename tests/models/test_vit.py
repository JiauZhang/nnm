import re

import torch
from conippets.config import Config
from transformers import ViTConfig, ViTModel

from nnm.models.vit import ViT


def map_state_dict(state_dict):
    mapped = {}
    for key, tensor in state_dict.items():
        if key.startswith("pooler."):
            continue
        key = re.sub(r"^vit\.", "", key)
        key = key.replace("embeddings.patch_embeddings.projection", "embeddings.patch_embeddings.proj")
        key = key.replace("encoder.layer", "layers")
        key = key.replace(".layernorm_before", ".norm1").replace(".layernorm_after", ".norm2")
        key = key.replace(".attention.attention.query", ".attn.q_proj")
        key = key.replace(".attention.attention.key", ".attn.k_proj")
        key = key.replace(".attention.attention.value", ".attn.v_proj")
        key = key.replace(".attention.output.dense", ".attn.o_proj")
        key = key.replace(".intermediate.dense", ".mlp.fc1")
        key = key.replace(".output.dense", ".mlp.fc2")
        key = re.sub(r"^layernorm\.", "norm.", key)
        mapped[key] = tensor
    return mapped


def tiny_config():
    return Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=32,
        patch_size=8,
        num_channels=3,
        layer_norm_eps=1e-12,
        hidden_act="gelu",
    )


def test_vit_matches_transformers():
    config = tiny_config()
    hf_config = ViTConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_channels=config.num_channels,
        layer_norm_eps=config.layer_norm_eps,
        hidden_act=config.hidden_act,
        qkv_bias=True,
    )
    torch.manual_seed(0)
    hf_model = ViTModel(hf_config, add_pooling_layer=False).eval()
    nnm_model = ViT(config).eval()
    nnm_model.load_state_dict(map_state_dict(hf_model.state_dict()), strict=True)
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = hf_model(pixels).last_hidden_state
        actual = nnm_model(pixels)
    assert torch.allclose(actual, expected, atol=1e-5)
