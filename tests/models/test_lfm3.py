import pytest
import torch

from conippets.config import Config
from nnm.cache import Cache
from nnm.models.lfm3 import Lfm3LM

TINY = {
    "hidden_size": 64,
    "head_dim": 16,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 4,
    "vocab_size": 100,
    "max_position_embeddings": 64,
    "sliding_window": 8,
    "tie_word_embeddings": True,
    "norm_eps": 1e-5,
    "conv_L_cache": 7,
    "conv_bias": False,
    "use_sandwich_norm": True,
    "pad_token_id": 1,
    "layer_types": ["conv", "sliding_attention", "full_attention", "conv"],
    "rope_parameters": {
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
    },
}


def tiny_config(**overrides):
    values = dict(TINY)
    values.update(overrides)
    return Config(**values)


class TestLfm3:
    def test_config_drives_the_architecture(self, weight_config):
        model = Lfm3LM(weight_config("LFM3-1.7B-Base"))
        assert sum(parameter.numel() for parameter in model.parameters()) == 1728801536

    @pytest.mark.parametrize(
        "layer_types", [["conv"] * 4, ["conv", "sliding_attention", "full_attention", "conv"], ["full_attention"] * 4]
    )
    def test_incremental_decoding_matches_the_full_pass(self, layer_types):
        config = tiny_config(layer_types=layer_types)
        model = Lfm3LM(config).eval()
        cache = Cache.from_layer_types(
            layer_types, capacity=64, window_size=config.sliding_window, state_size=config.conv_L_cache
        )
        input_ids = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            full = model(input_ids)
            chunks = [model(input_ids[:, :4], cache=cache), model(input_ids[:, 4:7], cache=cache)]
            chunks.append(model(input_ids[:, 7:], cache=cache))

        assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-4)
