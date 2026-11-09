import pytest
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForTextRecognition

from nnm.models.pp_ocrv5.mobile_det import PPOCRv5MobileDetectionModel
from nnm.models.pp_ocrv5.mobile_rec import PPOCRv5MobileRecognitionModel


def load_weights(model, repo, weight_config, weight_module):
    config = weight_config(repo)
    mapping = weight_module(repo)
    return config, mapping.map_state_dict(model.state_dict(), config)


def test_pp_ocrv5_mobile_det(reference_path, weight_config, weight_module):
    if reference_path is None:
        pytest.skip("--reference-path required")
    model = AutoModelForObjectDetection.from_pretrained(reference_path, device_map="auto")
    image_processor = AutoImageProcessor.from_pretrained(reference_path)
    config, state_dict = load_weights(model, "PP-OCRv5_mobile_det", weight_config, weight_module)
    nnm_model = PPOCRv5MobileDetectionModel(config)
    nnm_model.load_state_dict(state_dict)

    url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(model.device)
    test_input = inputs["pixel_values"]

    with torch.no_grad():
        original_logits = model(test_input)["last_hidden_state"]

    nnm_model = nnm_model.to(model.device).eval()
    with torch.no_grad():
        new_logits = nnm_model(test_input)

    assert original_logits.shape == new_logits.shape
    assert original_logits.cpu().numpy() == pytest.approx(new_logits.cpu().numpy(), abs=1e-4)


def test_pp_ocrv5_mobile_rec(reference_path, weight_config, weight_module):
    if reference_path is None:
        pytest.skip("--reference-path required")
    model = AutoModelForTextRecognition.from_pretrained(reference_path, device_map="auto")
    image_processor = AutoImageProcessor.from_pretrained(reference_path)
    config, state_dict = load_weights(model, "PP-OCRv5_mobile_rec", weight_config, weight_module)
    nnm_model = PPOCRv5MobileRecognitionModel(config)
    nnm_model.load_state_dict(state_dict)

    url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(model.device)
    test_input = inputs["pixel_values"]

    with torch.no_grad():
        original_logits = model(test_input)["last_hidden_state"]

    nnm_model = nnm_model.to(model.device).eval()
    with torch.no_grad():
        new_logits = nnm_model(test_input)

    assert original_logits.shape == new_logits.shape
    assert original_logits.cpu().numpy() == pytest.approx(new_logits.cpu().numpy(), abs=1e-4)
