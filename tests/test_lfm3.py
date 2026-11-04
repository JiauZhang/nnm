import copy

import pytest
import torch

from conippets.config import Config
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


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        base = AutoConfig.from_pretrained("LiquidAI/LFM3-1.7B-Base", trust_remote_code=True)
        config = copy.deepcopy(base)
        for key, value in {
            **{key: value for key, value in TINY.items() if key != "rope_parameters"},
            "pad_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "use_cache": False,
        }.items():
            setattr(config, key, value)
        config._attn_implementation = "eager"
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.float32).eval()
    except Exception as error:
        pytest.skip(f"LFM3 reference model is unavailable: {error}")


class TestLfm3:
    def test_config_drives_the_architecture(self, weight_config):
        model = Lfm3LM(weight_config("LFM3-1.7B-Base"))
        assert sum(parameter.numel() for parameter in model.parameters()) == 1728801536

    @pytest.mark.parametrize(
        "layer_types", [["conv"] * 4, ["conv", "sliding_attention", "full_attention", "conv"], ["full_attention"] * 4]
    )
    def test_incremental_decoding_matches_the_full_pass(self, layer_types):
        model = Lfm3LM(tiny_config(use_cache=True, layer_types=layer_types)).eval()
        input_ids = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            full = model(input_ids)
            model.clear_cache()
            chunks = [model(input_ids[:, :4]), model(input_ids[:, 4:7]), model(input_ids[:, 7:])]

        assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-4)

    def test_matches_the_upstream_model(self, reference):
        model = Lfm3LM(tiny_config(use_cache=False)).eval()
        state = {
            key.replace("model.", "backbone.", 1): value
            for key, value in reference.state_dict().items()
            if not key.startswith("lm_head.")
        }
        model.load_state_dict(state, strict=False)

        input_ids = torch.randint(0, 100, (2, 9))
        with torch.no_grad():
            expected = reference(input_ids).logits
            actual = model(input_ids)

        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
