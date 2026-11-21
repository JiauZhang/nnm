import glob
import os

import pytest
import torch

from nnm.models.depthanything2 import DepthAnythingV2
from tests.pretrained.isolated_upstream import run_isolated_upstream

pytestmark = pytest.mark.pretrained

HF_REPO = "depth-anything/Depth-Anything-V2-Small"

WORKER = """
import json
import sys
import types

import torch

package = types.ModuleType("depth_anything_v2")
package.__path__ = [REFERENCE.rstrip("/") + "/depth_anything_v2"]
sys.modules["depth_anything_v2"] = package

config = json.load(open(CONFIG))
kwargs = {
    "encoder": config.get("variant", "vits"),
    "features": config.get("features", 256),
    "out_channels": config.get("out_channels", [256, 512, 1024, 1024]),
    "use_bn": False,
    "use_clstoken": config.get("use_clstoken", False),
}
source = REFERENCE.rstrip("/") + "/depth_anything_v2/dpt.py"
if config.get("metric_max_depth"):
    kwargs["max_depth"] = config["metric_max_depth"]
    source = REFERENCE.rstrip("/") + "/metric_depth/depth_anything_v2/dpt.py"
    code = open(source).read().replace("from .", "from depth_anything_v2.")
    namespace = {"__name__": "depth_anything_v2_metric_dpt"}
    exec(compile(code, source, "exec"), namespace)
    upstream = namespace["DepthAnythingV2"]
else:
    from depth_anything_v2.dpt import DepthAnythingV2 as upstream

model = upstream(**kwargs).eval()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
x = torch.load(INPUT, weights_only=True)
with torch.no_grad():
    torch.save(model(x), sys.argv[-1])
"""


def upstream_weights(model_path):
    paths = glob.glob(os.path.join(model_path, "*.pth"))
    if not paths:
        hub = pytest.importorskip("huggingface_hub")
        snapshot = hub.snapshot_download(HF_REPO, allow_patterns=["*.pth"])
        paths = glob.glob(os.path.join(snapshot, "*.pth"))
    return paths[0]


def test_pretrained_loads(require_model_path, weight_limit):
    weight_limit(require_model_path)
    model = DepthAnythingV2.from_pretrained(require_model_path).eval()
    with torch.no_grad():
        depth = model(torch.randn(1, 3, 518, 518))
    assert tuple(depth.shape) == (1, 518, 518)
    assert torch.isfinite(depth).all()


def test_matches_upstream(require_model_path, require_reference_path, weight_limit, tmp_path):
    pytest.importorskip("cv2")
    weight_limit(require_model_path)
    model = DepthAnythingV2.from_pretrained(require_model_path).eval()

    torch.manual_seed(1)
    x = torch.randn(1, 3, 518, 518)
    input_path = tmp_path / "input.pt"
    torch.save(x, input_path)

    expected = run_isolated_upstream(
        WORKER,
        {
            "REFERENCE": require_reference_path,
            "CONFIG": os.path.join(require_model_path, "config.json"),
            "WEIGHTS": upstream_weights(require_model_path),
            "INPUT": input_path,
        },
        tmp_path / "expected.pt",
    )

    with torch.no_grad():
        depth = model(x)
    assert torch.allclose(depth, expected, atol=1e-4)
