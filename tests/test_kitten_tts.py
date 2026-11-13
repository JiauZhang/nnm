import json
import os
import sys

import numpy as np
import pytest
import torch
from conippets.config import Config

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


def _repo_variant(model_path):
    manifest = json.loads(open(os.path.join(model_path, "nnm_weights.json")).read())
    key = next(iter(manifest["modules"]))
    variant = key[len("kitten_tts_") :] if key.startswith("kitten_tts_") else None
    return manifest["modules"][key], variant


def test_pretrained_loads(model_path, weight_limit):
    if model_path is None:
        pytest.skip("--model-path required")
    weight_limit(model_path)
    entry, variant = _repo_variant(model_path)
    model = KittenTTS.from_pretrained(model_path, variant=variant).eval()
    config = Config.from_json(os.path.join(model_path, entry["config"]))
    out, _, _ = _infer(model, config.n_token)
    assert out["waveform"].numel() > 0


def test_matches_upstream(model_path, reference_path, weight_limit):
    if model_path is None or reference_path is None:
        pytest.skip("--model-path and --reference-path required")
    weight_limit(model_path)
    pytest.importorskip("scipy")
    sys.path.insert(0, reference_path)
    from kittenml.kittentts2_tiny.torch_model import TorchModel  # noqa: PLC0415

    entry, variant = _repo_variant(model_path)
    config = Config.from_json(os.path.join(model_path, entry["config"]))
    model = KittenTTS.from_pretrained(model_path, variant=variant).eval()

    source = entry["sources"][0]
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    weights_path = hf_hub_download(source["repo"], source["pattern"])
    arch = json.loads(open(os.path.join(model_path, entry["config"])).read())
    reference = TorchModel(arch, weights_path, device="cpu")

    out, ids, style = _infer(model, config.n_token)
    torch.manual_seed(42)
    ref_wav, ref_dur = reference(ids[0].tolist(), style.numpy(), 1.0)

    np.testing.assert_array_equal(out["duration"].numpy(), ref_dur)
    wav = out["waveform"].numpy()
    assert len(wav) == len(ref_wav)
    np.testing.assert_allclose(wav, ref_wav, atol=1e-4, rtol=1e-4)
