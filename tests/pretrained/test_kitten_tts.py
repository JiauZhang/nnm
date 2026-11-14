import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conippets.config import Config
from huggingface_hub import hf_hub_download

from nnm.models.kitten_tts import KittenTTS

pytestmark = pytest.mark.pretrained


def infer(model, n_token, seed=1234):
    torch.manual_seed(seed)
    ids = torch.randint(0, n_token, (1, 24))
    style = torch.randn(1, 256)
    torch.manual_seed(42)
    return model.infer(ids, style, speed=1.0), ids, style


def repo_variant(model_path):
    manifest = json.loads((Path(model_path) / "nnm_weights.json").read_text())
    key = next(iter(manifest["modules"]))
    variant = key[len("kitten_tts_") :] if key.startswith("kitten_tts_") else None
    return manifest["modules"][key], variant


def test_pretrained_loads(require_model_path, weight_limit):
    weight_limit(require_model_path)
    entry, variant = repo_variant(require_model_path)
    model = KittenTTS.from_pretrained(require_model_path, variant=variant).eval()
    config = Config.from_json(str(Path(require_model_path) / entry["config"]))
    out, _, _ = infer(model, config.n_token)
    assert out["waveform"].numel() > 0


def test_matches_upstream(require_model_path, require_reference_path, weight_limit):
    weight_limit(require_model_path)
    pytest.importorskip("scipy")
    sys.path.insert(0, require_reference_path)
    from kittenml.kittentts2_tiny.torch_model import TorchModel  # noqa: PLC0415

    entry, variant = repo_variant(require_model_path)
    config = Config.from_json(str(Path(require_model_path) / entry["config"]))
    model = KittenTTS.from_pretrained(require_model_path, variant=variant).eval()

    source = entry["sources"][0]
    weights_path = hf_hub_download(source["repo"], source["pattern"])
    arch = json.loads((Path(require_model_path) / entry["config"]).read_text())
    reference = TorchModel(arch, weights_path, device="cpu")

    out, ids, style = infer(model, config.n_token)
    torch.manual_seed(42)
    ref_wav, ref_dur = reference(ids[0].tolist(), style.numpy(), 1.0)

    np.testing.assert_array_equal(out["duration"].numpy(), ref_dur)
    wav = out["waveform"].numpy()
    assert len(wav) == len(ref_wav)
    np.testing.assert_allclose(wav, ref_wav, atol=1e-4, rtol=1e-4)
