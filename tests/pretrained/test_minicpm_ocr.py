import pytest
import torch
from conippets.config import Config
from nnm.models.minicpm_ocr.model import MiniCPMOcrModel


pytestmark = pytest.mark.pretrained

VISION_TINY = {
    "hidden_size": 16,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 4,
    "patch_size": 2,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
    "window_kernel_size": [2, 2],
}

TEXT_TINY = {
    "vocab_size": 100,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "hidden_act": "silu",
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 4,
    "linear_value_head_dim": 4,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 2,
    "attention_bias": False,
    "pad_token_id": 0,
    "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
}


def tiny_config():
    return Config(
        vision_config=VISION_TINY,
        text_config=TEXT_TINY,
        insert_layer_id=1,
        image_token_id=90,
        video_token_id=91,
        downsample_mode="4x",
        merge_kernel_size=[2, 2],
        merger_times=1,
        tie_word_embeddings=True,
    )


def packed_pixels(target_sizes, patch_size, seed=0):
    num_patches = int((target_sizes[:, 0] * target_sizes[:, 1]).sum())
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, patch_size, num_patches * patch_size, generator=generator)


@pytest.fixture(scope="module")
def upstream(weight_module):
    try:
        from transformers.models.minicpm_ocr.configuration_minicpm_ocr import (
            MiniCPMOcrConfig,
            MiniCPMOcrVisionConfig,
        )
        from transformers.models.minicpm_ocr.modeling_minicpm_ocr import MiniCPMOcrForConditionalGeneration
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    except Exception as error:
        pytest.skip(f"MiniCPM-OCR reference model is unavailable: {error}")

    config = MiniCPMOcrConfig(
        vision_config=MiniCPMOcrVisionConfig(**VISION_TINY),
        text_config=Qwen3_5TextConfig(**TEXT_TINY),
        insert_layer_id=1,
        image_token_id=90,
        video_token_id=91,
        downsample_mode="4x",
        merge_kernel_size=(2, 2),
        merger_times=1,
        tie_word_embeddings=True,
    )
    reference = MiniCPMOcrForConditionalGeneration(config).eval()
    model = MiniCPMOcrModel(tiny_config()).eval()

    state = {key: value for key, value in reference.state_dict().items() if not key.startswith("lm_head.")}
    model.load_state_dict(weight_module("MiniCPM-OCR-1.0").map_state_dict(state, model.config), strict=True)
    return model, reference


class TestUpstreamParity:
    @pytest.mark.parametrize("downsample_mode,target_sizes", [("4x", [[4, 4], [2, 2]]), ("16x", [[4, 4]])])
    @pytest.mark.pretrained
    def test_image_features_match_upstream(self, upstream, downsample_mode, target_sizes):
        model, reference = upstream
        sizes = torch.tensor(target_sizes, dtype=torch.int32)
        pixel_values = packed_pixels(sizes, 2)

        with torch.no_grad():
            actual = model.get_image_features(pixel_values, sizes, downsample_mode=downsample_mode)
            expected = reference.get_image_features(pixel_values, sizes, downsample_mode=downsample_mode).pooler_output

        assert [feature.shape for feature in actual] == [feature.shape for feature in expected]
        for feature, reference_feature in zip(actual, expected):
            torch.testing.assert_close(feature, reference_feature, atol=1e-5, rtol=1e-5)

    @pytest.mark.pretrained
    def test_logits_match_upstream(self, upstream):
        model, reference = upstream
        target_sizes = torch.tensor([[4, 4], [2, 2]], dtype=torch.int32)
        pixel_values = packed_pixels(target_sizes, 2)
        input_ids = torch.tensor([[1, 2, 90, 90, 90, 90, 90, 3, 4]])

        with torch.no_grad():
            actual = model(input_ids, pixel_values=pixel_values, target_sizes=target_sizes)
            expected = reference(input_ids, pixel_values=pixel_values, target_sizes=target_sizes).logits

        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.pretrained
    def test_from_pretrained_loads_upstream_checkpoint(self, upstream, tmp_path):
        import json

        from safetensors.torch import save_file

        model, reference = upstream
        state = {key: value.contiguous() for key, value in reference.state_dict().items() if not key.startswith("lm_head.")}
        save_file(state, tmp_path / "model.safetensors")
        config = tiny_config()
        config["weight_pattern"] = "model.safetensors"
        (tmp_path / "config.json").write_text(json.dumps(config, indent=2))

        loaded = MiniCPMOcrModel.from_pretrained(str(tmp_path))
        for key, value in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)
