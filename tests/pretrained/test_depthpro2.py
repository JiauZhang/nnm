import glob
import os

import pytest
import torch

from nnm.models.depthpro2 import DepthPro2, FOVEstimator
from tests.pretrained.isolated_upstream import run_isolated_upstream

pytestmark = pytest.mark.pretrained

HF_REPO = "apple/DepthPro2-S"
FOV_HF_REPO = "apple/DepthPro2-focal"
WEIGHT_FILE = "model.safetensors"

STUBS = """
import sys
import types

if "av" not in sys.modules:
    sys.modules["av"] = types.ModuleType("av")
if "pillow_heif" not in sys.modules:
    pillow_heif = types.ModuleType("pillow_heif")
    pillow_heif.register_heif_opener = lambda: None
    sys.modules["pillow_heif"] = pillow_heif
sys.path.insert(0, REFERENCE + "/src")
"""

DEPTH_WORKER = (
    STUBS
    + """
import torch

from depth_pro_2.model import PRESETS, DepthPro2 as UpDepthPro2

preset = PRESETS["depthpro2-s"]
model = UpDepthPro2(
    backbone_preset=preset.backbone_preset,
    reassembled_blocks=preset.reassembled_blocks,
    cross_frame_blocks=preset.cross_frame_blocks,
    dims_reassemble=preset.dims_reassemble,
    dim_decoder=preset.dim_decoder,
    fov=False,
    precision=torch.float32,
).eval()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
x = torch.load(INPUT, weights_only=True)
with torch.no_grad():
    torch.save(model.predict_inverse_depth(x), sys.argv[-1])
"""
)

FOV_WORKER = (
    STUBS
    + """
import torch

from depth_pro_2.network.fov import FOVEstimator as UpFOVEstimator

model = UpFOVEstimator(precision=torch.float32).eval()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
x = torch.load(INPUT, weights_only=True)
with torch.no_grad():
    torch.save(model(x), sys.argv[-1])
"""
)


def upstream_weights_path(model_path):
    paths = glob.glob(os.path.join(model_path, WEIGHT_FILE))
    if not paths:
        hub = pytest.importorskip("huggingface_hub")
        snapshot = hub.snapshot_download(HF_REPO, allow_patterns=[WEIGHT_FILE])
        paths = glob.glob(os.path.join(snapshot, WEIGHT_FILE))
    return paths[0]


def forward(model, size=(96, 128)):
    torch.manual_seed(1)
    x = torch.randn(1, 3, *size)
    with torch.no_grad():
        return x, model.predict_inverse_depth(x)


def test_pretrained_loads(require_model_path, weight_limit):
    weight_limit(require_model_path)
    model = DepthPro2.from_pretrained(require_model_path).eval()
    _, depth = forward(model)
    assert tuple(depth.shape) == (1, 96, 128)
    assert torch.isfinite(depth).all()


def test_matches_upstream(require_model_path, require_reference_path, weight_limit, tmp_path):
    pytest.importorskip("timm")
    weight_limit(require_model_path)
    model = DepthPro2.from_pretrained(require_model_path).eval()
    x, depth = forward(model)
    input_path = tmp_path / "input.pt"
    torch.save(x, input_path)

    expected = run_isolated_upstream(
        DEPTH_WORKER,
        {"REFERENCE": require_reference_path, "WEIGHTS": upstream_weights_path(require_model_path), "INPUT": input_path},
        tmp_path / "expected.pt",
    )
    assert torch.allclose(depth, expected, atol=1e-5)


def test_multiframe_matches_upstream(require_model_path, require_reference_path, weight_limit, tmp_path):
    pytest.importorskip("timm")
    weight_limit(require_model_path)
    model = DepthPro2.from_pretrained(require_model_path).eval()
    torch.manual_seed(2)
    x = torch.randn(3, 3, 64, 96)
    input_path = tmp_path / "input.pt"
    torch.save(x, input_path)

    expected = run_isolated_upstream(
        DEPTH_WORKER,
        {"REFERENCE": require_reference_path, "WEIGHTS": upstream_weights_path(require_model_path), "INPUT": input_path},
        tmp_path / "expected.pt",
    )
    with torch.no_grad():
        depth = model.predict_inverse_depth(x)
    assert torch.allclose(depth, expected, atol=1e-5)


def fov_weights_path(model_path):
    paths = glob.glob(os.path.join(model_path, WEIGHT_FILE))
    if not paths:
        hub = pytest.importorskip("huggingface_hub")
        snapshot = hub.snapshot_download(FOV_HF_REPO, allow_patterns=[WEIGHT_FILE])
        paths = glob.glob(os.path.join(snapshot, WEIGHT_FILE))
    return paths[0]


def test_fov_loads(local_weights_root, weight_limit):
    fov_path = os.path.join(local_weights_root, "depth-pro-2-focal")
    weight_limit(fov_path)
    model = FOVEstimator.from_pretrained(fov_path).eval()
    torch.manual_seed(3)
    with torch.no_grad():
        fov = model(torch.randn(1, 3, 256, 384))
    assert tuple(fov.shape) == (1, 2)
    assert torch.isfinite(fov).all()


def test_fov_matches_upstream(local_weights_root, require_reference_path, weight_limit, tmp_path):
    pytest.importorskip("timm")
    fov_path = os.path.join(local_weights_root, "depth-pro-2-focal")
    weight_limit(fov_path)
    model = FOVEstimator.from_pretrained(fov_path).eval()
    torch.manual_seed(3)
    x = torch.randn(1, 3, 256, 384)
    input_path = tmp_path / "input.pt"
    torch.save(x, input_path)

    expected = run_isolated_upstream(
        FOV_WORKER,
        {"REFERENCE": require_reference_path, "WEIGHTS": fov_weights_path(fov_path), "INPUT": input_path},
        tmp_path / "expected.pt",
    )
    with torch.no_grad():
        fov = model(x)
    assert torch.allclose(fov, expected, atol=1e-4)
