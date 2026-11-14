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


class TestLfm3Vl:
    def test_config_drives_the_architecture(self, weight_config):
        config = weight_config("LFM3-VL-2B")
        with torch.device("meta"):
            model = Lfm3VlModel(config)
        assert sum(parameter.numel() for parameter in model.parameters()) == 1918181888
