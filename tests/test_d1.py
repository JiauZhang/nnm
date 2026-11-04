import copy

import pytest
import torch

from conippets.config import Config
from nnm.models.lfm3 import D1OmniModel, D1VlModel, Lfm3OmniModel, pointer_probs, position_logits, state_attention_mask
from test_lfm3_omni import TINY as OMNI_TINY
from test_lfm3_omni import tiny_inputs as omni_inputs

TEXT = {
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
}

VISION = {
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
}

VL_TINY = {
    "model_type": "lfm3_vl",
    "text_config": TEXT,
    "vision_config": VISION,
    "downsample_factor": 2,
    "projector_hidden_size": 48,
    "projector_bias": True,
    "projector_hidden_act": "gelu",
    "projector_layer_norm_eps": 1e-6,
    "image_token_id": 92,
    "video_token_id": 93,
    "tie_word_embeddings": True,
    "d1_codes": 2,
    "d1_attention": "set",
    "d1_set_size": 4,
    "d1_numbers": True,
}


def vl_config():
    return Config(**copy.deepcopy(VL_TINY))


def vl_inputs():
    input_ids = torch.randint(0, 80, (2, 12))
    input_ids[:, 4:8] = 92
    pixel_values = torch.randn(32, 96)
    image_grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    return input_ids, pixel_values, image_grid


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        repo = "LiquidAI/d1-2B"
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        for key, value in TEXT.items():
            setattr(config.text_config, key, value)
        for key, value in VISION.items():
            setattr(config.vision_config, key, value)
        for key, value in VL_TINY.items():
            if not isinstance(value, dict):
                setattr(config, key, value)
        config._attn_implementation = "sdpa"
        config.dtype = "float32"
        model = get_class_from_dynamic_module(config.auto_map["AutoModel"], repo)
        return model(config).to(torch.float32).eval()
    except Exception as error:
        pytest.skip(f"D1 reference model is unavailable: {error}")


class TestD1:
    @pytest.mark.parametrize(
        "model_class, repo, expected",
        [
            (D1VlModel, "d1-2B", 1918181888),
            (D1OmniModel, "d1-Omni-2B", 2191713280),
        ],
    )
    def test_config_drives_the_architecture(self, model_class, repo, expected, weight_config):
        with torch.device("meta"):
            model = model_class(weight_config(repo))
        assert sum(parameter.numel() for parameter in model.parameters()) == expected

    def test_d1_adds_no_parameters(self):
        with torch.device("meta"):
            vl = D1VlModel(vl_config())
            omni = D1OmniModel(Config(**copy.deepcopy(OMNI_TINY)))
            plain_omni = Lfm3OmniModel(Config(**copy.deepcopy(OMNI_TINY)))
        assert sum(p.numel() for p in vl.parameters()) == 128512
        assert sum(p.numel() for p in omni.parameters()) == sum(p.numel() for p in plain_omni.parameters())

    def test_vl_forward(self):
        model = D1VlModel(vl_config()).eval()
        input_ids, pixel_values, image_grid = vl_inputs()
        with torch.no_grad():
            logits = model(input_ids, pixel_values, image_grid)
        assert logits.shape == (2, 12, 100)

    def test_omni_forward(self):
        model = D1OmniModel(Config(**copy.deepcopy(OMNI_TINY))).eval()
        input_ids, input_features, feature_mask, codes, pixel_values, image_grid = omni_inputs()
        with torch.no_grad():
            logits = model(input_ids, input_features, feature_mask, codes, pixel_values, image_grid)
        assert logits.shape == (1, 24, 100)

    def test_checkpoint_keys_strip_the_model_prefix(self, weight_config, weight_module):
        for repo, model_class in (("d1-2B", D1VlModel), ("d1-Omni-2B", D1OmniModel)):
            config = weight_config(repo)
            model = model_class(vl_config() if model_class is D1VlModel else Config(**copy.deepcopy(OMNI_TINY)))
            source = {f"model.{key}": None for key in model.state_dict()}
            assert set(weight_module(repo).map_state_dict(source, config)) == set(model.state_dict())

    def test_matches_the_upstream_model(self, reference):
        model = D1VlModel(vl_config()).eval()
        state = {key[len("model.") :]: value for key, value in reference.state_dict().items() if not key.startswith("lm_head.")}
        model.load_state_dict(state, strict=True)

        input_ids, pixel_values, image_grid = vl_inputs()
        with torch.no_grad():
            expected = reference(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid,
                attention_mask=torch.ones_like(input_ids),
            ).logits
            actual = model(input_ids, pixel_values, image_grid)
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


class TestD1Utilities:
    def test_state_mask_reads_the_state_both_ways(self):
        mask = state_attention_mask(torch.tensor([3, 3], dtype=torch.long), valid=torch.ones(2, 5, dtype=torch.bool))
        assert mask.shape == (2, 1, 5, 5)
        allowed = mask[0, 0] == 0
        assert allowed[0].tolist() == [True, True, True, False, False]
        assert allowed[3].tolist() == [True, True, True, True, False]

    def test_set_mask_keeps_each_question_to_itself(self):
        questions = torch.tensor([[-1, -1, -1, 1, 2], [-1, -1, -1, 1, 2]])
        mask = state_attention_mask(torch.tensor([3, 3]), question_ids=questions, valid=torch.ones(2, 5, dtype=torch.bool))
        allowed = (mask[0, 0] == 0).tolist()
        assert allowed[3] == [True, True, True, True, False]
        assert allowed[4] == [True, True, True, False, True]
        assert allowed[0] == [True, True, True, True, True]

    def test_padding_columns_are_never_read(self):
        valid = torch.tensor([[True, True, False]])
        mask = state_attention_mask(torch.tensor([2], dtype=torch.long), valid=valid)
        assert torch.equal(mask[0, 0, :, 2], torch.tensor([float("-inf"), float("-inf"), float("-inf")]))

    def test_pointer_reads_positions(self):
        hidden = torch.randn(2, 3, 8)
        states = torch.randn(2, 5, 8)
        valid = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        logits = position_logits(hidden, states, valid)
        probs = pointer_probs(hidden, states, valid)
        assert logits.shape == (2, 3, 5)
        assert logits[1, :, 3:].isinf().all() and (logits[1, :, 3:] < 0).all()
        assert torch.allclose(probs.sum(-1), torch.ones(2, 3))
