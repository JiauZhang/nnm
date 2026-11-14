import torch
from conippets.config import Config
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from nnm.models.perception_encoder import PerceptionEncoderVision


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


def test_vision_matches_dinov3():
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
        layerscale_init=1e-5,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        ffn_bias=True,
        num_register_tokens=2,
    )
    torch.manual_seed(0)
    reference = DINOv3ViTModel(DINOv3ViTConfig(**config)).eval()
    model = PerceptionEncoderVision(config).eval()
    state_dict = map_state_dict(reference.state_dict())
    state_dict["rope_embed.periods"] = model.rope_embed.periods
    model.load_state_dict(state_dict, strict=True)
    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = reference(pixel_values).last_hidden_state
        features = model.forward_features(pixel_values)
    assert torch.allclose(features["x_norm_clstoken"], expected[:, 0], atol=1e-5)
    assert torch.allclose(features["x_storage_tokens"], expected[:, 1:3], atol=1e-5)
    assert torch.allclose(features["x_norm_patchtokens"], expected[:, 3:], atol=1e-5)
