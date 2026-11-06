import torch
from conippets.config import Config
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from nnm.models.perception_encoder import PerceptionEncoder, PerceptionEncoderVision


def tiny_config():
    return Config(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=8,
        in_chans=3,
        n_storage_tokens=2,
        norm_eps=1e-6,
        layerscale_init=1e-5,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        use_text=True,
        text_hidden_size=32,
        text_num_heads=4,
        text_depth=2,
        text_intermediate_size=64,
        text_vocab_size=100,
        text_rope_base=10000.0,
        projection_dim=32,
    )


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
        layerscale_value=1e-5,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        mlp_bias=True,
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


def test_perception_encoder_forward():
    torch.manual_seed(0)
    model = PerceptionEncoder(tiny_config()).eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 100, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask)
    assert output["x_norm_patchtokens"].shape == (2, 16, 32)
    assert output["x_storage_tokens"].shape == (2, 2, 32)
    assert output["logits_per_image"].shape == (2, 2)


def test_perception_encoder_vision_only():
    model = PerceptionEncoder(tiny_config()).eval()
    with torch.no_grad():
        features = model.vision_model.forward_features(torch.randn(1, 3, 32, 32))
    assert features["x_norm_clstoken"].shape == (1, 32)
