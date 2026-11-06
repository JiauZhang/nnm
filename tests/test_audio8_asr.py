import math

import pytest
import torch
from conippets.config import Config

from nnm.models.audio8_asr import Audio8ASR


def tiny_config():
    return Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_key_value_groups=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        pad_token_id=0,
        audio_token_id=5,
        tie_word_embeddings=True,
        use_cache=False,
        sliding_window=None,
        qwen3_asr_mlp_tower_layers=1,
        qwen3_asr_mlp_tower_hidden_size=0,
        qwen3_asr_mlp_tower_dropout=0.0,
        qwen3_asr_audio_config=dict(
            num_mel_bins=8,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=32,
            n_window=2,
            n_window_infer=4,
            conv_chunksize=5,
            downsample_hidden_size=8,
            output_dim=32,
            activation_function="gelu",
            activation_dropout=0.0,
            attention_dropout=0.0,
            dropout=0.0,
        ),
    )


def make_inputs(audio_token_id=5, num_mel_bins=8, frames=16):
    input_ids = torch.tensor([[0, audio_token_id, audio_token_id, audio_token_id, 6, 7]])
    input_features = torch.randn(1, num_mel_bins, frames)
    feature_lens = torch.tensor([frames])
    return input_ids, input_features, feature_lens


def test_audio8_forward():
    torch.manual_seed(0)
    model = Audio8ASR(tiny_config()).eval()
    input_ids, input_features, feature_lens = make_inputs()
    with torch.no_grad():
        logits = model(input_ids, input_features=input_features, feature_lens=feature_lens)
        text_only = model(input_ids)
    assert logits.shape == (1, 6, 100)
    assert not torch.allclose(logits, text_only)


def test_audio8_pretrained_matches_reference(model_path, reference_path):
    if model_path is None or reference_path is None:
        pytest.skip("--model-path and --reference-path required")
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    reference = AutoModelForCausalLM.from_pretrained(reference_path, trust_remote_code=True, torch_dtype=torch.float32).eval()
    reference.language_model.lm_head.weight = reference.language_model.model.embed_tokens.weight
    positional = reference.audio_encoder.positional_embedding.positional_embedding
    hidden_size = positional.shape[1]
    increment = math.log(10000) / (hidden_size // 2 - 1)
    inverse = torch.exp(-increment * torch.arange(hidden_size // 2, dtype=torch.float32))
    angles = torch.arange(positional.shape[0], dtype=torch.float32)[:, None] * inverse[None]
    positional.copy_(torch.cat([angles.sin(), angles.cos()], dim=1))
    model = Audio8ASR.from_pretrained(model_path).float().eval()
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
    assert torch.allclose(actual, expected, atol=1e-4)
