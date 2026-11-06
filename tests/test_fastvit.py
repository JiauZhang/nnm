import pytest
import torch
from conippets.config import Config

from nnm.models.fastvit import FastViT


def _reference_model():
    pytest.importorskip("timm")
    from timm.models.fastvit import FastVit  # noqa: PLC0415

    return FastVit


def tiny_config():
    return Config(
        in_chans=3,
        layers=[2, 2],
        embed_dims=[16, 32],
        token_mixers=["repmixer", "repmixer"],
        mlp_ratios=[4.0, 4.0],
        downsamples=[False, True],
        num_classes=10,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
    )


def test_fastvit_matches_reference():
    torch.manual_seed(0)
    reference = _reference_model()(
        in_chans=3,
        layers=(2, 2),
        embed_dims=(16, 32),
        token_mixers=("repmixer", "repmixer"),
        mlp_ratios=(4.0, 4.0),
        downsamples=(False, True),
        num_classes=10,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
    ).eval()
    model = FastViT(tiny_config()).eval()
    state_dict = {}
    for key, value in reference.state_dict().items():
        key = key.replace("head.fc.", "classifier.")
        key = key.replace(".se.fc1.", ".se.conv1.").replace(".se.fc2.", ".se.conv2.")
        state_dict[key] = value
    model.load_state_dict(state_dict, strict=True)
    pixel_values = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = reference(pixel_values)
        actual = model(pixel_values)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_fastvit_features():
    model = FastViT(tiny_config()).eval()
    with torch.no_grad():
        features = model.forward_features(torch.randn(1, 3, 64, 64))
    assert features.shape == (1, 64, 8, 8)
