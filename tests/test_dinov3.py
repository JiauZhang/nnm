import torch
from conippets.config import Config
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from nnm.models.dinov3 import DINOv3


def map_state_dict(state_dict):
    mapped = {}
    for key, tensor in state_dict.items():
        if key.startswith("rope_embeddings."):
            continue
        key = key.replace("embeddings.patch_embeddings.", "patch_embed.proj.")
        key = key.replace("embeddings.cls_token", "cls_token")
        key = key.replace("embeddings.mask_token", "mask_token")
        key = key.replace("embeddings.register_tokens", "storage_tokens")
        key = key.replace("model.layer.", "layers.")
        key = key.replace(".attention.", ".attn.")
        key = key.replace(".layer_scale1.lambda1", ".ls1.gamma")
        key = key.replace(".layer_scale2.lambda1", ".ls2.gamma")
        key = key.replace(".mlp.up_proj", ".mlp.fc1")
        key = key.replace(".mlp.down_proj", ".mlp.fc2")
        mapped[key] = tensor
    return mapped


def test_dinov3_matches_transformers():
    config = Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=32,
        patch_size=8,
        num_channels=3,
        layer_norm_eps=1e-6,
        rope_theta=100.0,
        layerscale_value=1e-5,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        mlp_bias=True,
        num_register_tokens=2,
    )
    hf_config = DINOv3ViTConfig(**config)
    torch.manual_seed(0)
    hf_model = DINOv3ViTModel(hf_config).eval()
    nnm_model = DINOv3(config).eval()
    state_dict = map_state_dict(hf_model.state_dict())
    state_dict["rope_embed.periods"] = nnm_model.rope_embed.periods
    nnm_model.load_state_dict(state_dict, strict=True)
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = hf_model(pixels).last_hidden_state
        actual = nnm_model(pixels)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_dinov3_token_layout():
    config = Config(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=8,
        image_size=32,
        num_register_tokens=4,
        layerscale_init=1e-5,
        ffn_layer="mlp",
        ffn_ratio=4.0,
    )
    model = DINOv3(config).eval()
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        hidden = model(pixels)
    assert hidden.shape == (2, 1 + 4 + 16, 32)
    features = model.forward_features(pixels)
    assert features["x_norm_clstoken"].shape == (2, 32)
    assert features["x_storage_tokens"].shape == (2, 4, 32)
    assert features["x_norm_patchtokens"].shape == (2, 16, 32)
