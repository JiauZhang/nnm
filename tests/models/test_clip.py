import torch
from conippets.config import Config
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from nnm.models.clip import CLIP, CLIPVisionModel

VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 32,
    "patch_size": 8,
    "layer_norm_eps": 1e-6,
    "hidden_act": "quick_gelu",
}

TEXT_CONFIG = {
    "vocab_size": 100,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 16,
    "layer_norm_eps": 1e-6,
    "hidden_act": "quick_gelu",
}


def make_hf_config():
    text_config = CLIPTextConfig(**TEXT_CONFIG, eos_token_id=99, bos_token_id=0, pad_token_id=0)
    return CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=CLIPVisionConfig(**VISION_CONFIG).to_dict(),
        projection_dim=16,
        logit_scale_init_value=2.6592,
    )


def make_nnm_config():
    return Config(
        vision_config=VISION_CONFIG,
        text_config=TEXT_CONFIG,
        projection_dim=16,
        logit_scale_init_value=2.6592,
    )


def map_state_dict(state_dict):
    mapped = {}
    for key, tensor in state_dict.items():
        if key.startswith("vision_model.encoder.layers."):
            key = key.replace(".layer_norm1", ".norm1").replace(".layer_norm2", ".norm2")
            key = key.replace(".self_attn.out_proj", ".attn.o_proj").replace(".self_attn.", ".attn.")
        elif key.startswith("text_model.encoder.layers."):
            key = key.replace(".self_attn.out_proj", ".self_attn.o_proj")
            key = key.replace(".layer_norm1", ".input_layernorm").replace(".layer_norm2", ".post_attention_layernorm")
        mapped[key] = tensor
    return mapped


def make_inputs():
    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(3, 99, (2, 8))
    input_ids[:, -1] = 99
    return pixel_values, input_ids, torch.ones_like(input_ids)


def test_clip_matches_transformers():
    torch.manual_seed(0)
    reference = CLIPModel(make_hf_config()).eval()
    model = CLIP(make_nnm_config()).eval()
    model.load_state_dict(map_state_dict(reference.state_dict()), strict=True)

    pixel_values, input_ids, attention_mask = make_inputs()
    with torch.no_grad():
        expected = reference(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        actual = model(pixel_values, input_ids, attention_mask)

    for name in ("logits_per_image", "logits_per_text", "image_embeds", "text_embeds"):
        assert torch.allclose(actual[name], getattr(expected, name), atol=1e-5)


def test_vision_patch_embeds_entry():
    model = CLIPVisionModel(Config(**VISION_CONFIG)).eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    patch_embeds = model.embeddings.patch_embedding(pixel_values)
    with torch.no_grad():
        expected = model(pixel_values=pixel_values)
        actual = model(patch_embeds=patch_embeds)

    assert actual["last_hidden_state"].shape == (2, 17, 32)
    assert torch.allclose(actual["last_hidden_state"], expected["last_hidden_state"], atol=1e-6)
    assert torch.allclose(actual["pooler_output"], expected["pooler_output"], atol=1e-6)


def test_vision_post_layernorm_can_be_disabled():
    model = CLIPVisionModel(Config(**VISION_CONFIG), post_layernorm=False)
    assert model.post_layernorm is None
    assert "post_layernorm.weight" not in model.state_dict()
