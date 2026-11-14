import pytest
import torch

from nnm.models.kitten_tts import KittenTTS

REPOS = {
    "pico": "kitten-tts-2-pico",
    "nano": "kitten-tts-2-nano",
    "mini": "kitten-tts-2-mini",
}


def _infer(model, n_token, seed=1234):
    torch.manual_seed(seed)
    ids = torch.randint(0, n_token, (1, 24))
    style = torch.randn(1, 256)
    torch.manual_seed(42)
    return model.infer(ids, style, speed=1.0), ids, style


@pytest.mark.parametrize("family", list(REPOS))
def test_architecture(family, weight_config):
    config = weight_config(REPOS[family])
    model = KittenTTS(config).eval()
    out, _, _ = _infer(model, config.n_token)
    assert out["waveform"].dim() == 1
    assert out["waveform"].numel() > 0
    assert out["duration"].shape == (24,)
    assert (out["duration"] >= 1).all()


@pytest.mark.parametrize("family", list(REPOS))
def test_speed_shortens(family, weight_config):
    config = weight_config(REPOS[family])
    model = KittenTTS(config).eval()
    slow, ids, style = _infer(model, config.n_token)
    torch.manual_seed(42)
    fast = model.infer(ids, style, speed=2.0)
    assert fast["duration"].sum() <= slow["duration"].sum()
