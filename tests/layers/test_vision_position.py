import pytest
import torch

from nnm.layers.vision.position import merge_position_ids


def test_position_ids_follow_merge_blocks():
    grid = torch.tensor([[1, 4, 4], [2, 4, 8]])
    positions = merge_position_ids(grid, 2)
    assert positions.shape == (16 + 64, 2)
    assert positions[:16, 0].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
    assert torch.equal(positions[16:48], positions[48:])

    vision_utils = pytest.importorskip("transformers.vision_utils")

    assert torch.equal(positions, vision_utils.get_vision_position_ids(grid, 2))
