import copy

import pytest
import torch

from conippets.config import Config
from nnm.models.lfm3.vision import Lfm3VisionTower, vision_position_ids

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


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        repo = "LiquidAI/LFM3-Omni-2B"
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        for key, value in VISION.items():
            setattr(config.vision_config, key, value)
        model = get_class_from_dynamic_module("modeling_lfm3_vl.Lfm3VlVisionModel", repo)
        return model(config.vision_config).eval()
    except Exception as error:
        pytest.skip(f"LFM3-VL vision reference is unavailable: {error}")


def test_position_ids_follow_merge_blocks():
    grid = torch.tensor([[1, 4, 4], [2, 4, 8]])
    positions = vision_position_ids(grid, 2)
    assert positions.shape == (16 + 64, 2)
    assert positions[:16, 0].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
    assert torch.equal(positions[16:48], positions[48:])

    vision_utils = pytest.importorskip("transformers.vision_utils")

    assert torch.equal(positions, vision_utils.get_vision_position_ids(grid, 2))


def test_mixed_frame_shapes_match_the_reference(reference):
    model = Lfm3VisionTower(Config(**copy.deepcopy(VISION))).eval()
    model.load_state_dict(reference.state_dict(), strict=True)

    grid = torch.tensor([[1, 4, 4], [1, 4, 8]])
    pixel_values = torch.randn(
        int(grid.prod(-1).sum()), VISION["in_channels"] * VISION["temporal_patch_size"] * VISION["patch_size"] ** 2
    )
    with torch.no_grad():
        expected = reference(pixel_values, grid).last_hidden_state
        actual = model(pixel_values, grid)

    assert actual.shape == (48, 32)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
