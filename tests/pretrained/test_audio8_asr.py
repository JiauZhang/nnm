import math

import pytest
import torch
from transformers import AutoModelForCausalLM

from nnm.models.audio8_asr import Audio8ASR

pytestmark = pytest.mark.pretrained


def make_inputs(audio_token_id=5, num_mel_bins=8, frames=16):
    input_ids = torch.tensor([[0, audio_token_id, audio_token_id, audio_token_id, 6, 7]])
    input_features = torch.randn(1, num_mel_bins, frames)
    feature_lens = torch.tensor([frames])
    return input_ids, input_features, feature_lens


def test_audio8_pretrained_matches_reference(require_model_path, require_reference_path):
    reference = AutoModelForCausalLM.from_pretrained(
        require_reference_path, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    reference.language_model.lm_head.weight = reference.language_model.model.embed_tokens.weight
    positional = reference.audio_encoder.positional_embedding.positional_embedding
    hidden_size = positional.shape[1]
    increment = math.log(10000) / (hidden_size // 2 - 1)
    inverse = torch.exp(-increment * torch.arange(hidden_size // 2, dtype=torch.float32))
    angles = torch.arange(positional.shape[0], dtype=torch.float32)[:, None] * inverse[None]
    positional.copy_(torch.cat([angles.sin(), angles.cos()], dim=1))

    model = Audio8ASR.from_pretrained(require_model_path).float().eval()
    audio_config = model.config.qwen3_asr_audio_config
    num_mel_bins = audio_config["num_mel_bins"] if isinstance(audio_config, dict) else audio_config.num_mel_bins
    input_ids, input_features, feature_lens = make_inputs(
        audio_token_id=model.audio_token_id,
        num_mel_bins=num_mel_bins,
        frames=32,
    )
    with torch.no_grad():
        expected = reference(input_ids=input_ids, input_features=input_features, feature_lens=feature_lens).logits
        actual = model(input_ids, input_features=input_features, feature_lens=feature_lens)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
