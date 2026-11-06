import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
from conippets.config import Config


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to the pretrained model directory",
    )
    parser.addoption(
        "--reference-path",
        action="store",
        default=None,
        help="Path to the upstream checkpoint a converted nnm model was built from",
    )
    parser.addoption(
        "--weights-root",
        action="store",
        default=None,
        help="Path to the local nnm weight repositories prepared for upload",
    )


@pytest.fixture(scope="module")
def model_path(request):
    return request.config.getoption("--model-path")


@pytest.fixture(scope="module")
def reference_path(request):
    return request.config.getoption("--reference-path")


@pytest.fixture(scope="session")
def weights_root(request):
    return request.config.getoption("--weights-root")


@pytest.fixture(scope="session")
def weight_config(weights_root):
    def load(repo):
        if weights_root is None:
            pytest.skip("--weights-root required")
        return Config.from_json(os.path.join(weights_root, repo, "config.json"))

    return load


@pytest.fixture(scope="session")
def weight_module(weights_root):
    def load(repo):
        if weights_root is None:
            pytest.skip("--weights-root required")
        path = os.path.join(weights_root, repo, "weight_mapping.py")
        spec = importlib.util.spec_from_file_location(f"weight_mapping_{repo.replace('/', '_')}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def _count_checkpoint_parameters(directory):
    from safetensors import safe_open  # noqa: PLC0415

    total = 0
    for path in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                total += int(math.prod(shape))
    return total


@pytest.fixture(scope="session")
def weight_limit():
    def check(target):
        if isinstance(target, (str, Path)):
            if not os.path.isdir(target):
                return
            total = _count_checkpoint_parameters(target)
        else:
            total = sum(parameter.numel() for parameter in target.parameters())
        if total > 2_000_000_000:
            pytest.skip(f"checkpoint has {total / 1e9:.2f}B parameters, weight verification is limited to 2B")

    return check


@pytest.fixture(scope="session")
def source_repo():
    def load(model_path):
        directory = model_path
        if not os.path.isdir(directory):
            from huggingface_hub import snapshot_download

            directory = snapshot_download(directory, allow_patterns=["nnm_weights.json"])
        manifest = json.loads(Path(directory, "nnm_weights.json").read_text())
        return manifest["sources"][0]["repo"]

    return load
