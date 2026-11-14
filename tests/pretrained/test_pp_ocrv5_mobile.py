import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForTextRecognition

from nnm.models.pp_ocrv5.mobile_det import PPOCRv5MobileDetectionModel
from nnm.models.pp_ocrv5.mobile_rec import PPOCRv5MobileRecognitionModel

pytestmark = pytest.mark.pretrained

CASES = [
    ("PP-OCRv5_mobile_det", PPOCRv5MobileDetectionModel, AutoModelForObjectDetection),
    ("PP-OCRv5_mobile_rec", PPOCRv5MobileRecognitionModel, AutoModelForTextRecognition),
]


def demo_image():
    pixels = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("repo, nnm_cls, reference_cls", CASES)
def test_matches_reference(repo, nnm_cls, reference_cls, require_reference_path, weight_config, weight_module):
    reference = reference_cls.from_pretrained(require_reference_path, device_map="auto")
    processor = AutoImageProcessor.from_pretrained(require_reference_path)
    config = weight_config(repo)
    mapping = weight_module(repo)

    model = nnm_cls(config)
    model.load_state_dict(mapping.map_state_dict(model.state_dict(), config))
    model = model.to(reference.device).eval()

    pixel_values = processor(images=demo_image(), return_tensors="pt").to(reference.device)["pixel_values"]
    with torch.no_grad():
        expected = reference(pixel_values)["last_hidden_state"]
        actual = model(pixel_values)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
