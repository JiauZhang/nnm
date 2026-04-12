import pytest, requests, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForTextRecognition
from nnm.models.pp_ocrv5.mobile_det import PPOCRv5MobileDetectionModel
from nnm.models.pp_ocrv5.mobile_rec import PPOCRv5MobileRecognitionModel

def test_pp_ocrv5_mobile_det():
    model_path = "./PP-OCRv5_mobile_det_safetensors"
    model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    nnm_model = PPOCRv5MobileDetectionModel()
    nnm_model.load_hf_state_dict(model.state_dict())

    url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(model.device)
    test_input = inputs["pixel_values"]

    with torch.no_grad():
        original_output = model(test_input)
        original_logits = original_output["last_hidden_state"]

    nnm_model = nnm_model.to(model.device)
    nnm_model.eval()
    with torch.no_grad():
        new_logits = nnm_model(test_input)

    assert original_logits.shape == new_logits.shape
    assert original_logits.cpu().numpy() == pytest.approx(new_logits.cpu().numpy(), abs=1e-4)

def test_pp_ocrv5_mobile_rec():
    model_path="./PP-OCRv5_mobile_rec_safetensors"
    model = AutoModelForTextRecognition.from_pretrained(model_path, device_map="auto")
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    nnm_model = PPOCRv5MobileRecognitionModel()
    nnm_model.load_hf_state_dict(model.state_dict())

    image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png", stream=True).raw).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(model.device)
    test_input = inputs["pixel_values"]

    with torch.no_grad():
        original_output = model(test_input)
        original_logits = original_output["last_hidden_state"]

    nnm_model = nnm_model.to(model.device)
    nnm_model.eval()
    with torch.no_grad():
        new_logits = nnm_model(test_input)

    assert original_logits.shape == new_logits.shape
    assert original_logits.cpu().numpy() == pytest.approx(new_logits.cpu().numpy(), abs=1e-4)
