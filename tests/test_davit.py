import torch
from conippets.config import Config

from nnm.models.davit import DaViT


def tiny_config():
    return Config(
        image_size=64,
        in_channels=3,
        embed_dim=[16, 32],
        depths=[1, 1],
        num_heads=[1, 2],
        num_groups=[1, 2],
        patch_size=[7, 3],
        patch_stride=[4, 2],
        patch_padding=[3, 1],
        patch_prenorm=[False, True],
        window_size=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
    )


def test_davit_forward():
    torch.manual_seed(0)
    model = DaViT(tiny_config()).eval()
    with torch.no_grad():
        features = model(torch.randn(2, 3, 64, 64))
    assert features.shape == (2, 32, 8, 8)


def test_davit_multiscale_features():
    model = DaViT(tiny_config()).eval()
    with torch.no_grad():
        _, features = model.forward_features(torch.randn(1, 3, 64, 64))
    assert len(features) == 2
    assert features[0].shape == (1, 16, 16, 16)
    assert features[1].shape == (1, 32, 8, 8)
