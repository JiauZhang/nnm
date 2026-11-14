import torch

from nnm.models.qwen3_asr.audio import feat_extract_lengths


def test_feature_length_reduction():
    assert feat_extract_lengths(torch.tensor([100])).item() == 13
    assert feat_extract_lengths(torch.tensor([150])).item() == 20
