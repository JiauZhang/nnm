import torch
from conippets.config import Config

from nnm.models.dino import DINOv2
from nnm.models.dpt import FeatureFusionBlock2d


def make_config(fov=False):
    return Config(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        num_channels=3,
        patch_size=8,
        image_size=32,
        layer_norm_eps=1e-6,
        num_register_tokens=2,
        qkv_bias=True,
        proj_bias=True,
        hidden_act="gelu",
    )


def test_dinov2_forward_shape():
    model = DINOv2(make_config()).eval()
    with torch.no_grad():
        hidden = model(torch.randn(2, 3, 32, 32))
    assert tuple(hidden.shape) == (2, 19, 32)


def test_dinov2_intermediate_layers():
    model = DINOv2(make_config()).eval()
    with torch.no_grad():
        taps = model.get_intermediate_layers(torch.randn(1, 3, 32, 32), [0, 2], return_class_token=True)
    assert len(taps) == 2
    for patches, cls in taps:
        assert tuple(patches.shape) == (1, 16, 32)
        assert tuple(cls.shape) == (1, 32)


def test_dinov2_masked_positions():
    model = DINOv2(make_config()).eval()
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 0] = True
    with torch.no_grad():
        hidden = model(torch.randn(2, 3, 32, 32), bool_masked_pos=mask)
    assert torch.isfinite(hidden).all()


def test_dinov2_swiglu_ffn():
    from nnm.layers.mlp import GatedMLP

    config = make_config()
    config.ffn_layer = "swiglu8"
    config.ffn_ratio = 2.0
    model = DINOv2(config).eval()
    assert isinstance(model.blocks[0].mlp, GatedMLP)
    with torch.no_grad():
        hidden = model(torch.randn(1, 3, 32, 32))
    assert torch.isfinite(hidden).all()


def test_dpt_decoder_upsample_hook():
    class BilinearFusion(FeatureFusionBlock2d):
        def make_upsample(self, dim):
            return torch.nn.Identity()

        def upsample(self, x, size=None):
            return torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

    block = BilinearFusion(16, with_skip=True).eval()
    x = torch.randn(1, 16, 4, 4)
    skip = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        out = block(x, skip)
    assert tuple(out.shape) == (1, 16, 8, 8)
    assert torch.isfinite(out).all()
