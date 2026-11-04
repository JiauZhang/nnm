import copy

import pytest
import torch

from conippets.config import Config
from nnm.models.lfm3 import Lfm3OmniModel

TINY = {
    "model_type": "lfm3_omni",
    "text_config": {
        "hidden_size": 64,
        "head_dim": 16,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 100,
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "tie_word_embeddings": True,
        "norm_eps": 1e-5,
        "conv_L_cache": 7,
        "conv_bias": False,
        "use_sandwich_norm": True,
        "pad_token_id": 1,
        "use_cache": False,
        "layer_types": ["conv", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
        },
    },
    "audio_encoder_config": {
        "num_mel_bins": 8,
        "conv_channels": 8,
        "hidden_size": 32,
        "sliding_window": 4,
        "text_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 3,
            "vocab_size": 2,
            "max_position_embeddings": 64,
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "conv_bias": False,
            "use_cache": False,
            "pad_token_id": 0,
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "layer_types": ["conv", "full_attention", "full_attention"],
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        },
    },
    "depth_config": {
        "hidden_size": 32,
        "head_dim": 8,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 1,
        "max_position_embeddings": 16,
        "norm_eps": 1e-5,
        "conv_L_cache": 7,
        "conv_bias": False,
        "use_sandwich_norm": False,
        "use_cache": False,
        "pad_token_id": None,
        "layer_types": ["full_attention", "full_attention"],
        "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 1.0, "rope_type": "default"}},
    },
    "vision_config": {
        "model_type": "lfm3_vl_vision",
        "in_channels": 3,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": ["conv", "full_attention", "conv"],
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"},
        "norm_eps": 1e-5,
        "patch_size": 4,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "conv_kernel_size": 3,
    },
    "num_codebooks": 3,
    "codebook_size": 11,
    "audio_eos_code": 10,
    "audio_token_id": 90,
    "audio_out_token_id": 91,
    "image_token_id": 92,
    "video_token_id": 93,
    "downsample_factor": 2,
    "projector_hidden_size": 48,
    "projector_bias": True,
    "projector_hidden_act": "gelu",
    "projector_layer_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


def tiny_config():
    return Config(**copy.deepcopy(TINY))


def tiny_inputs():
    input_ids = torch.randint(0, 80, (1, 24))
    input_ids[0, 4:8] = 92
    input_ids[0, 8:18] = 90
    input_ids[0, 18] = 91
    input_features = torch.randn(1, 8, 78)
    feature_mask = torch.ones(1, 78, dtype=torch.bool)
    pixel_values = torch.randn(16, 96)
    image_grid = torch.tensor([[1, 4, 4]])
    codes = torch.randint(0, 11, (1, 3))
    return input_ids, input_features, feature_mask, codes, pixel_values, image_grid


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        repo = "LiquidAI/LFM3-Omni-2B"
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        for key, value in TINY["text_config"].items():
            setattr(config.text_config, key, value)
        for key, value in TINY["audio_encoder_config"]["text_config"].items():
            setattr(config.audio_encoder_config.text_config, key, value)
        for key in ("num_mel_bins", "conv_channels", "hidden_size", "sliding_window"):
            setattr(config.audio_encoder_config, key, TINY["audio_encoder_config"][key])
        for key, value in TINY["depth_config"].items():
            setattr(config.depth_config, key, value)
        for key, value in TINY["vision_config"].items():
            setattr(config.vision_config, key, value)
        for key, value in TINY.items():
            if not isinstance(value, dict):
                setattr(config, key, value)
        config._attn_implementation = "eager"
        model = get_class_from_dynamic_module(config.auto_map["AutoModelForMultimodalLM"], repo)
        return model(config).eval()
    except Exception as error:
        pytest.skip(f"LFM3-Omni reference model is unavailable: {error}")


def reference_state(model):
    return {key[len("model.") :]: value for key, value in model.state_dict().items() if not key.startswith("lm_head.")}


class TestLfm3Omni:
    def test_config_drives_the_architecture(self, weight_config):
        with torch.device("meta"):
            model = Lfm3OmniModel(weight_config("LFM3-Omni-2B"))
        assert sum(parameter.numel() for parameter in model.parameters()) == 2191713280

    def test_modality_features_and_logits_shapes(self):
        model = Lfm3OmniModel(tiny_config()).eval()
        input_ids, input_features, feature_mask, codes, pixel_values, image_grid = tiny_inputs()
        with torch.no_grad():
            audio = model.get_audio_features(input_features, feature_mask)
            image = model.get_image_features(pixel_values, image_grid)
            logits = model(input_ids, input_features, feature_mask, codes, pixel_values, image_grid)

        assert audio.shape == (10, 64)
        assert [tensor.shape for tensor in image] == [(4, 64)]
        assert logits.shape == (1, 24, 100)

    def test_all_placeholders_take_their_features(self):
        model = Lfm3OmniModel(tiny_config()).eval()
        input_ids, input_features, feature_mask, codes, pixel_values, image_grid = tiny_inputs()
        with torch.no_grad():
            audio = model.get_audio_features(input_features, feature_mask)
            image = torch.cat(model.get_image_features(pixel_values, image_grid))
            frames = model.get_audio_out_embeds(codes)
            embeds = model.embed_inputs(input_ids, input_features, feature_mask, codes, pixel_values, image_grid)
            tokens = model.language_model.embed_tokens(input_ids)

        masks = {
            "image": input_ids == 92,
            "audio": input_ids == 90,
            "frames": input_ids == 91,
        }
        placeholders = masks["image"] | masks["audio"] | masks["frames"]
        assert torch.allclose(embeds[masks["image"]], image)
        assert torch.allclose(embeds[masks["audio"]], audio)
        assert torch.allclose(embeds[masks["frames"]], frames)
        assert torch.allclose(embeds[~placeholders], tokens[~placeholders])

    def test_checkpoint_keys_strip_the_model_prefix(self, weight_config, weight_module):
        config = weight_config("LFM3-Omni-2B")
        model = Lfm3OmniModel(tiny_config())
        source = {f"model.{key}": None for key in model.state_dict()}
        mapped = weight_module("LFM3-Omni-2B").map_state_dict(source, config)
        assert set(mapped) == set(model.state_dict())

    def test_matches_the_upstream_model(self, reference):
        model = Lfm3OmniModel(tiny_config()).eval()
        model.load_state_dict(reference_state(reference), strict=True)

        input_ids, input_features, feature_mask, codes, pixel_values, image_grid = tiny_inputs()
        with torch.no_grad():
            expected = reference(
                input_ids=input_ids,
                input_features=input_features,
                feature_attention_mask=feature_mask,
                audio_out_codes=codes,
                pixel_values=pixel_values,
                image_grid_thw=image_grid,
                attention_mask=torch.ones_like(input_ids),
            ).logits
            actual = model(input_ids, input_features, feature_mask, codes, pixel_values, image_grid)

        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
