import copy

import pytest
import torch

from conippets.config import Config
from nnm.models.lfm3 import Lfm3AudioModel


pytestmark = pytest.mark.pretrained

TINY = {
    "model_type": "lfm3_audio",
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
        "pad_token_id": None,
        "layer_types": ["full_attention", "full_attention"],
        "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 1.0, "rope_type": "default"}},
    },
    "num_codebooks": 3,
    "codebook_size": 11,
    "audio_eos_code": 10,
    "audio_token_id": 90,
    "audio_out_token_id": 91,
    "tie_word_embeddings": True,
}


def tiny_config():
    return Config(**copy.deepcopy(TINY))


def tiny_inputs():
    input_ids = torch.randint(0, 80, (2, 16))
    input_ids[0, 4:14] = 90
    input_ids[0, 14] = 91
    input_ids[1, 4:8] = 90
    input_ids[1, 8] = 91
    input_features = torch.randn(2, 8, 78)
    feature_mask = torch.ones(2, 78, dtype=torch.bool)
    feature_mask[1, 30:] = False
    codes = torch.randint(0, 11, (2, 3))
    return input_ids, input_features, feature_mask, codes


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        repo = "LiquidAI/LFM3-Audio-2B"
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        for key, value in TINY["text_config"].items():
            setattr(config.text_config, key, value)
        for key, value in TINY["audio_encoder_config"]["text_config"].items():
            setattr(config.audio_encoder_config.text_config, key, value)
        for key in ("num_mel_bins", "conv_channels", "hidden_size", "sliding_window"):
            setattr(config.audio_encoder_config, key, TINY["audio_encoder_config"][key])
        for key, value in TINY["depth_config"].items():
            setattr(config.depth_config, key, value)
        for key, value in TINY.items():
            if not isinstance(value, dict):
                setattr(config, key, value)
        config._attn_implementation = "eager"
        model = get_class_from_dynamic_module(config.auto_map["AutoModelForMultimodalLM"], repo)
        return model(config).eval()
    except Exception as error:
        pytest.skip(f"LFM3-Audio reference model is unavailable: {error}")


def reference_state(model):
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("lm_head."):
            continue
        if key.startswith("model.language_model."):
            key = "backbone." + key[len("model.language_model.") :]
        else:
            key = key[len("model.") :]
        state[key] = value
    return state


class TestLfm3Audio:
    @pytest.mark.pretrained
    def test_matches_the_upstream_model(self, reference):
        model = Lfm3AudioModel(tiny_config()).eval()
        model.load_state_dict(reference_state(reference), strict=True)

        input_ids, input_features, feature_mask, codes = tiny_inputs()
        with torch.no_grad():
            expected = reference(
                input_ids=input_ids,
                input_features=input_features,
                feature_attention_mask=feature_mask,
                audio_out_codes=codes,
                attention_mask=torch.ones_like(input_ids),
            ).logits
            actual = model(input_ids, input_features, feature_mask, codes)

        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
