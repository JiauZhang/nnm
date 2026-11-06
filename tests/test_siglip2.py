import torch
from conippets.config import Config
from transformers import Siglip2TextConfig, Siglip2TextModel, Siglip2VisionConfig, Siglip2VisionModel

from nnm.models.siglip2 import SigLIP2, SigLIP2TextModel, SigLIP2VisionModel


def map_vision_state_dict(state_dict, hidden_size):
    mapped = {}
    for key, tensor in state_dict.items():
        key = key.removeprefix("vision_model.")
        if key.startswith("embeddings.patch_embedding."):
            key = key.replace("embeddings.patch_embedding.", "embeddings.patch_embedding.proj.")
            tensor = tensor.reshape(tensor.shape[0], 3, 8, 8) if tensor.ndim == 2 else tensor
        elif key.startswith("encoder.layers."):
            key = key.replace("encoder.layers.", "encoder.")
            key = key.replace(".layer_norm1", ".norm1").replace(".layer_norm2", ".norm2")
            key = key.replace(".self_attn.out_proj", ".attn.o_proj")
            key = key.replace(".self_attn.", ".attn.")
        elif key.startswith("head.attention.in_proj_weight"):
            query, key_, value = tensor.chunk(3, dim=0)
            mapped["head.attention.q_proj.weight"] = query
            mapped["head.attention.k_proj.weight"] = key_
            mapped["head.attention.v_proj.weight"] = value
            continue
        elif key.startswith("head.attention.in_proj_bias"):
            query, key_, value = tensor.chunk(3, dim=0)
            mapped["head.attention.q_proj.bias"] = query
            mapped["head.attention.k_proj.bias"] = key_
            mapped["head.attention.v_proj.bias"] = value
            continue
        elif key.startswith("head.attention.out_proj"):
            key = key.replace("head.attention.out_proj", "head.attention.o_proj")
        mapped[key] = tensor
    return mapped


def test_vision_matches_transformers():
    config = Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=32,
        patch_size=8,
        num_patches=256,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        vision_use_head=True,
    )
    torch.manual_seed(0)
    hf_model = Siglip2VisionModel(Siglip2VisionConfig(**config)).eval()
    nnm_model = SigLIP2VisionModel(config).eval()
    nnm_model.load_state_dict(map_vision_state_dict(hf_model.state_dict(), 32), strict=True)

    pixels = torch.randn(2, 3, 32, 32)
    patchified = pixels.reshape(2, 3, 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5).reshape(2, 16, -1)
    spatial_shapes = torch.tensor([[4, 4], [4, 4]])
    attention_mask = torch.ones(2, 32, 32)
    with torch.no_grad():
        expected = hf_model(patchified, attention_mask, spatial_shapes)
        actual = nnm_model(pixels)
    assert torch.allclose(actual["last_hidden_state"], expected.last_hidden_state, atol=1e-5)
    assert torch.allclose(actual["pooler_output"], expected.pooler_output, atol=1e-5)


def test_text_matches_transformers():
    config = Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
    )
    torch.manual_seed(0)
    hf_model = Siglip2TextModel(Siglip2TextConfig(**config)).eval()
    nnm_model = SigLIP2TextModel(config).eval()
    state_dict = {}
    for key, tensor in hf_model.state_dict().items():
        key = key.removeprefix("text_model.")
        key = key.replace(".self_attn.out_proj", ".self_attn.o_proj")
        key = key.replace(".layer_norm1", ".input_layernorm")
        key = key.replace(".layer_norm2", ".post_attention_layernorm")
        state_dict[key] = tensor
    nnm_model.load_state_dict(state_dict, strict=True)

    input_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        expected = hf_model(input_ids)
        actual = nnm_model(input_ids)
    assert torch.allclose(actual["last_hidden_state"], expected.last_hidden_state, atol=1e-5)
    assert torch.allclose(actual["pooler_output"], expected.pooler_output, atol=1e-5)


def tiny_config():
    return Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=32,
        patch_size=8,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        vision_use_head=True,
        vocab_size=100,
        max_position_embeddings=16,
    )


def test_siglip2_forward():
    torch.manual_seed(0)
    model = SigLIP2(tiny_config()).eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 100, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask)
    assert output["logits_per_image"].shape == (2, 2)
    assert output["logits_per_text"].shape == (2, 2)


def test_siglip2_vision_output():
    model = SigLIP2(tiny_config()).eval()
    with torch.no_grad():
        output = model.vision_model(torch.randn(1, 3, 32, 32))
    assert output["last_hidden_state"].shape == (1, 16, 32)
    assert output["pooler_output"].shape == (1, 32)
