import copy

import pytest
import torch

from conippets.config import Config
from nnm.models.lfm3 import Lfm3VlModel

TINY_VISION = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": ["conv", "full_attention", "conv", "full_attention"],
    "patch_size": 4,
    "temporal_patch_size": 1,
    "in_channels": 3,
    "spatial_merge_size": 2,
    "conv_kernel_size": 3,
    "norm_eps": 1e-5,
    "rope_parameters": {"rope_type": "axial", "rope_theta": 10000},
}

TINY_TEXT = {
    "hidden_size": 32,
    "head_dim": 8,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 3,
    "vocab_size": 100,
    "max_position_embeddings": 64,
    "sliding_window": 4,
    "tie_word_embeddings": True,
    "norm_eps": 1e-5,
    "conv_L_cache": 7,
    "conv_bias": False,
    "use_sandwich_norm": True,
    "use_cache": False,
    "pad_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 3,
    "layer_types": ["conv", "sliding_attention", "full_attention"],
    "rope_parameters": {
        "full_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
    },
}

TINY = {
    "vision_config": TINY_VISION,
    "text_config": TINY_TEXT,
    "projector_hidden_act": "gelu",
    "projector_hidden_size": 64,
    "projector_bias": True,
    "projector_layer_norm_eps": 1e-6,
    "downsample_factor": 2,
    "image_token_id": 90,
    "video_token_id": 91,
    "tie_word_embeddings": True,
}

GRID = torch.tensor([[1, 4, 4], [1, 4, 6]])
IMAGE_TOKENS = [4, 6]
NUM_PATCHES = int((GRID[:, 1] * GRID[:, 2] * GRID[:, 0]).sum())


def tiny_config(**overrides):
    values = dict(TINY)
    values.update(overrides)
    return Config(**values)


def pixel_values_for(grid):
    patch_dim = TINY_VISION["in_channels"] * TINY_VISION["temporal_patch_size"] * TINY_VISION["patch_size"] ** 2
    patches = grid.repeat_interleave(grid[:, 0], dim=0)[:, 1:]
    counts = patches[:, 0] * patches[:, 1]
    return torch.randn(int(counts.sum()), patch_dim)


def translated_state_dict(state):
    translated = {}
    for key, value in state.items():
        if key.startswith("lm_head."):
            continue
        key = key[len("model.") :] if key.startswith("model.") else key
        if key.startswith("language_model."):
            key = "backbone." + key[len("language_model.") :]
        translated[key] = value
    return translated


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText

        base = AutoConfig.from_pretrained("LiquidAI/LFM3-VL-2B", trust_remote_code=True)
        config = copy.deepcopy(base)
        for key, value in TINY_VISION.items():
            setattr(config.vision_config, key, value)
        for key, value in TINY_TEXT.items():
            setattr(config.text_config, key, value)
        for key, value in TINY.items():
            if key not in ("vision_config", "text_config"):
                setattr(config, key, value)
        for current in (config, config.vision_config, config.text_config):
            current._attn_implementation = "eager"
        return AutoModelForImageTextToText.from_config(config, trust_remote_code=True).to(torch.float32).eval()
    except Exception as error:
        pytest.skip(f"LFM3-VL reference model is unavailable: {error}")


class TestLfm3Vl:
    def test_config_drives_the_architecture(self, weight_config):
        config = weight_config("LFM3-VL-2B")
        with torch.device("meta"):
            model = Lfm3VlModel(config)
        assert sum(parameter.numel() for parameter in model.parameters()) == 1918181888

    def test_vision_projector_and_image_token_shapes(self):
        model = Lfm3VlModel(tiny_config()).eval()
        pixel_values = pixel_values_for(GRID)

        with torch.no_grad():
            hidden_states = model.vision_tower(pixel_values, GRID)
            merged = model.multi_modal_projector(hidden_states)
            features = model.get_image_features(pixel_values, GRID)

        assert hidden_states.shape == (NUM_PATCHES, TINY_VISION["hidden_size"])
        assert merged.shape == (sum(IMAGE_TOKENS), TINY_TEXT["hidden_size"])
        assert [feature.shape[0] for feature in features] == IMAGE_TOKENS

        input_ids = torch.randint(0, TINY["image_token_id"], (2, 18))
        input_ids[0, 5 : 5 + IMAGE_TOKENS[0]] = TINY["image_token_id"]
        input_ids[1, 5 : 5 + IMAGE_TOKENS[1]] = TINY["image_token_id"]
        with torch.no_grad():
            inputs_embeds = model.embed_inputs(input_ids, pixel_values, GRID)
            logits = model(input_ids, pixel_values, GRID)
            text_logits = model(input_ids)

        assert inputs_embeds.shape == (2, 18, TINY_TEXT["hidden_size"])
        assert logits.shape == text_logits.shape == (2, 18, TINY_TEXT["vocab_size"])

    def test_matches_the_upstream_model(self, reference):
        model = Lfm3VlModel(tiny_config()).to(torch.float32).eval()
        missing, unexpected = model.load_state_dict(translated_state_dict(reference.state_dict()), strict=False)
        assert not missing and not unexpected

        pixel_values = pixel_values_for(GRID)
        input_ids = torch.randint(0, TINY["image_token_id"], (2, 18))
        input_ids[0, 5 : 5 + IMAGE_TOKENS[0]] = TINY["image_token_id"]
        input_ids[1, 5 : 5 + IMAGE_TOKENS[1]] = TINY["image_token_id"]

        with torch.no_grad():
            expected_vision = reference.model.vision_tower(pixel_values, grid_thw=GRID).last_hidden_state
            actual_vision = model.vision_tower(pixel_values, GRID)
            expected_features = reference.model.multi_modal_projector(expected_vision)
            actual_features = model.multi_modal_projector(actual_vision)
            expected_logits = reference(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=GRID).logits
            actual_logits = model(input_ids, pixel_values, GRID)
            expected_text = reference(input_ids=input_ids).logits
            actual_text = model(input_ids)

        assert torch.allclose(actual_vision, expected_vision, atol=1e-4, rtol=1e-4)
        assert torch.allclose(actual_features, expected_features, atol=1e-4, rtol=1e-4)
        assert torch.allclose(actual_logits, expected_logits, atol=1e-4, rtol=1e-4)
        assert torch.allclose(actual_text, expected_text, atol=1e-4, rtol=1e-4)
