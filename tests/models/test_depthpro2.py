import torch
from conippets.config import Config

from nnm.models.depthpro2 import DepthPro2, FOVEstimator, CrossFrameCache


def make_config(fov=False):
    return Config(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        num_register_tokens=2,
        layer_norm_eps=1e-6,
        patch_size=8,
        rope_theta=100.0,
        pos_embed_rope_normalize_coords="max",
        layerscale_init=1e-5,
        qkv_bias=False,
        proj_bias=True,
        hidden_act="gelu",
        fov=fov,
        reassembled_blocks=[0, 1, 2, 3],
        cross_frame_blocks=[1, 3],
        dims_reassemble=[8, 16, 32, 32],
        dim_decoder=16,
    )


def test_forward_single_frame_shape():
    model = DepthPro2(make_config()).eval()
    with torch.no_grad():
        out = model.predict_inverse_depth(torch.randn(1, 3, 32, 32))
    assert tuple(out.shape) == (1, 32, 32)
    assert torch.isfinite(out).all()


def test_forward_pads_to_multiple():
    model = DepthPro2(make_config()).eval()
    with torch.no_grad():
        out = model.predict_inverse_depth(torch.randn(1, 3, 20, 28))
    assert tuple(out.shape) == (1, 20, 28)


def test_forward_multi_frame_shape():
    model = DepthPro2(make_config()).eval()
    with torch.no_grad():
        out = model.predict_inverse_depth(torch.randn(3, 3, 32, 32))
    assert tuple(out.shape) == (3, 32, 32)


def test_cross_frame_frames_attend_to_each_other():
    model = DepthPro2(make_config()).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features, grid = model.encoder.encode(x)
        swapped, _ = model.encoder.encode(x.flip(0))
    assert not torch.allclose(features[0][0], swapped[0][0], atol=1e-6)


def test_cache_matches_buffered_single_frame():
    model = DepthPro2(make_config()).eval()
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        buffered = model.predict_inverse_depth(x)
        cache = CrossFrameCache(window=3)
        cached = torch.stack([model.predict_inverse_depth(x[i : i + 1], cache)[0] for i in range(3)])
    assert torch.allclose(buffered, cached, atol=1e-5)


def test_infer_with_known_focal():
    model = DepthPro2(make_config()).eval()
    image = torch.randn(3, 32, 32)
    with torch.no_grad():
        result = model.infer(image, f_px=100.0)
    assert tuple(result["inverse_depth"].shape) == (32, 32)
    assert tuple(result["depth"].shape) == (32, 32)
    assert result["focallength_px"].item() == 100.0
    assert torch.allclose(result["depth"], 100.0 / (32 * result["inverse_depth"]), atol=1e-5)


def test_infer_no_focal_gives_none_depth():
    model = DepthPro2(make_config()).eval()
    with torch.no_grad():
        result = model.infer(torch.randn(3, 32, 32))
    assert result["depth"] is None
    assert result["focallength_px"] is None


def test_stream_buffered_returns_all_frames():
    model = DepthPro2(make_config()).eval()
    frames = [torch.randn(3, 32, 32) for _ in range(5)]
    stream = model.stream(window=3, overlap=1, f_px=100.0, mode="buffered")
    results = []
    for frame in frames:
        results.extend(stream.step(frame))
    results.extend(stream.flush())
    assert len(results) == 5
    for result in results:
        assert tuple(result["depth"].shape) == (32, 32)


def test_stream_unbuffered_returns_all_frames():
    model = DepthPro2(make_config()).eval()
    frames = [torch.randn(3, 32, 32) for _ in range(5)]
    stream = model.stream(window=3, f_px=100.0, mode="unbuffered")
    results = []
    for frame in frames:
        results.extend(stream.step(frame))
    results.extend(stream.flush())
    assert len(results) == 5


def test_stream_rejects_invalid_mode():
    model = DepthPro2(make_config()).eval()
    try:
        model.stream(mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid mode")


def test_kv_cache_ring_overwrites():
    cache = CrossFrameCache(window=2)
    key = torch.randn(1, 2, 4, 4)
    value = torch.randn(1, 2, 4, 4)
    pooled_key = torch.randn(1, 2, 1, 4)
    pooled_value = torch.randn(1, 2, 1, 4)
    for _ in range(3):
        cache.update_prefix(0, torch.randn(1, 1, 4))
        keys, values = cache.update(0, key, value, pooled_key, pooled_value)
    assert keys.shape[-2] == 1 + 2 * 4


def make_fov_config():
    config = make_config()
    config.num_hidden_layers = 12
    return config


def test_fov_forward_shape():
    model = FOVEstimator(make_fov_config()).eval()
    with torch.no_grad():
        fov = model(torch.randn(1, 3, 64, 64))
    assert tuple(fov.shape) == (1, 2)
    assert (fov > 0).all() and (fov < 180).all()


def test_fov_focal_length():
    model = FOVEstimator(make_fov_config()).eval()
    with torch.no_grad():
        fpx = model.focal_length_px(torch.randn(2, 3, 64, 64))
    assert fpx.ndim == 0
    assert fpx.item() > 0


def test_fov_resize_crops_wide_frames():
    model = FOVEstimator(make_fov_config()).eval()
    resized = model.resize(torch.randn(1, 3, 64, 256))
    h, w = resized.shape[-2:]
    assert w / h <= 1.7778 + 1e-6


def test_backbone_uses_nnm_mlp_and_swiglu():
    from nnm.layers.mlp import GatedMLP, MLP

    plain = DepthPro2(make_config()).eval()
    assert isinstance(plain.encoder.blocks[0].mlp, MLP)

    swiglu_config = make_config()
    swiglu_config.ffn_layer = "swiglu64"
    swiglu_config.ffn_ratio = 2.0
    swiglu = DepthPro2(swiglu_config).eval()
    assert isinstance(swiglu.encoder.blocks[0].mlp, GatedMLP)
    with torch.no_grad():
        out = swiglu.predict_inverse_depth(torch.randn(1, 3, 32, 32))
    assert tuple(out.shape) == (1, 32, 32)
