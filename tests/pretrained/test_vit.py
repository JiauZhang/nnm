import pytest
import torch

from nnm.models.vit import ViT

pytestmark = pytest.mark.pretrained


def test_vit_pretrained(require_model_path, weight_limit):
    weight_limit(require_model_path)
    model = ViT.from_pretrained(require_model_path).eval()
    pixels = torch.zeros(1, 3, 224, 224)
    pixels[:, :, 100:140, 60:100] = 1.0
    with torch.no_grad():
        hidden = model(pixels)
    assert hidden.shape == (1, 197, 768)
