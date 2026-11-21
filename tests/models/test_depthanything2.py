import torch
from conippets.config import Config

from nnm.models.depthanything2 import DepthAnythingV2


def make_config(metric_max_depth=0):
    return Config(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        num_channels=3,
        patch_size=8,
        image_size=32,
        layer_norm_eps=1e-6,
        layerscale_init=1e-5,
        num_register_tokens=0,
        qkv_bias=True,
        proj_bias=True,
        hidden_act="gelu",
        features=16,
        out_channels=[8, 16, 32, 32],
        use_clstoken=False,
        tapped_blocks=[0, 1, 2, 3],
        variant="vits",
        metric_max_depth=metric_max_depth,
    )


def test_forward_relative_depth():
    model = DepthAnythingV2(make_config()).eval()
    with torch.no_grad():
        depth = model(torch.randn(2, 3, 32, 32))
    assert tuple(depth.shape) == (2, 32, 32)
    assert (depth >= 0).all()
    assert torch.isfinite(depth).all()


def test_forward_metric_depth():
    model = DepthAnythingV2(make_config(metric_max_depth=10.0)).eval()
    with torch.no_grad():
        depth = model(torch.randn(2, 3, 32, 32))
    assert tuple(depth.shape) == (2, 32, 32)
    assert (depth >= 0).all() and (depth <= 10.0).all()


def test_use_clstoken_branch():
    config = make_config()
    config.use_clstoken = True
    model = DepthAnythingV2(config).eval()
    assert len(model.head.readout_projects) == 4
    with torch.no_grad():
        depth = model(torch.randn(1, 3, 32, 32))
    assert tuple(depth.shape) == (1, 32, 32)
    assert torch.isfinite(depth).all()


def test_variant_tap_indices():
    from nnm.models.depthanything2 import TAP_INDICES

    assert TAP_INDICES["vits"] == (2, 5, 8, 11)
    assert TAP_INDICES["vitl"] == (4, 11, 17, 23)
    config = make_config()
    del config["tapped_blocks"]
    config.variant = "vitl"
    model = DepthAnythingV2(config).eval()
    assert model.tapped_blocks == (4, 11, 17, 23)


def test_uses_shared_fusion_blocks():
    from nnm.models.dpt import BilinearFusionBlock

    model = DepthAnythingV2(make_config()).eval()
    for fusion in model.head.refinenets:
        assert isinstance(fusion, BilinearFusionBlock)


def test_metric_variants_load(local_weights_root):
    for repo in ["depth-anything-v2-metric-hypersim-small", "depth-anything-v2-metric-vkitti-small"]:
        model = DepthAnythingV2.from_pretrained(local_weights_root / repo).eval()
        with torch.no_grad():
            depth = model(torch.randn(1, 3, 518, 518))
        assert tuple(depth.shape) == (1, 518, 518)
        assert 0 <= depth.max() <= model.metric_max_depth
