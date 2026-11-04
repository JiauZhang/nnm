import copy

import pytest
import torch

from conippets.config import Config
from nnm.models.lfm3 import Lfm3AudioModel

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
    def test_config_drives_the_architecture(self, weight_config):
        with torch.device("meta"):
            model = Lfm3AudioModel(weight_config("LFM3-Audio-2B"))
        assert sum(parameter.numel() for parameter in model.parameters()) == 2002332928

    def test_audio_features_and_logits_shapes(self):
        model = Lfm3AudioModel(tiny_config()).eval()
        input_ids, input_features, feature_mask, codes = tiny_inputs()
        with torch.no_grad():
            features = model.get_audio_features(input_features, feature_mask)
            out_embeds = model.get_audio_out_embeds(codes)
            logits = model(input_ids, input_features, feature_mask, codes)
            depth = model.depth_decoder(torch.randn(2, 64), codes, torch.zeros(2, 3, dtype=torch.bool))

        assert features.shape == (14, 64)
        assert out_embeds.shape == (2, 64)
        assert depth.shape == (2, 3, 11)
        assert logits.shape == (2, 16, 100)

    def test_speech_placeholders_take_the_projected_features(self):
        model = Lfm3AudioModel(tiny_config()).eval()
        input_ids, input_features, feature_mask, codes = tiny_inputs()
        with torch.no_grad():
            features = model.get_audio_features(input_features, feature_mask)
            frame_embeds = model.get_audio_out_embeds(codes)
            embeds = model.embed_inputs(input_ids, input_features, feature_mask, codes)
            tokens = model.backbone.embed_tokens(input_ids)

        audio_positions = input_ids == 90
        out_positions = input_ids == 91
        assert torch.allclose(embeds[audio_positions], features)
        assert torch.allclose(embeds[out_positions], frame_embeds)
        assert torch.allclose(embeds[~(audio_positions | out_positions)], tokens[~(audio_positions | out_positions)])

    def test_checkpoint_keys_map_to_the_model(self, weight_config, weight_module):
        config = weight_config("LFM3-Audio-2B")
        model = Lfm3AudioModel(tiny_config())
        source = {}
        for key in model.state_dict():
            if key.startswith("backbone."):
                source["lfm.model." + key[len("backbone.") :]] = None
            elif key.startswith("audio_encoder.downsample."):
                source["encoder.downsample." + key[len("audio_encoder.downsample.") :]] = None
            elif key.startswith("audio_encoder.lfm."):
                source["encoder.lfm.lfm." + key[len("audio_encoder.lfm.") :]] = None
            elif key.startswith("audio_projector.layer_norm."):
                source["mlp.model.0." + key[len("audio_projector.layer_norm.") :]] = None
            elif key.startswith("audio_projector.linear_1."):
                source["mlp.model.1." + key[len("audio_projector.linear_1.") :]] = None
            elif key.startswith("audio_projector.linear_2."):
                source["mlp.model.3." + key[len("audio_projector.linear_2.") :]] = None
            elif key.startswith("depth_decoder.depth_linear."):
                source["depth_linear." + key[len("depth_decoder.depth_linear.") :]] = None
            elif key.startswith("depth_decoder.mask_embedding"):
                source["mask_embedding"] = None
            elif key.startswith("depth_decoder.model."):
                source["depthformer.model." + key[len("depth_decoder.model.") :]] = None
            elif key.startswith("depth_decoder.depth_embeddings."):
                source["depth_embeddings." + key[len("depth_decoder.depth_embeddings.") :]] = None
            else:
                source[key] = None

        mapped = weight_module("LFM3-Audio-2B").map_state_dict(source, config)
        assert set(mapped) == set(model.state_dict())

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
