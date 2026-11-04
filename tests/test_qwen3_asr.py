import numpy as np
import pytest
import torch

from conippets.config import Config
from nnm.models.qwen3_asr import Qwen3ASR, Qwen3ForcedAligner
from nnm.models.qwen3_asr.audio import feat_extract_lengths

ASR = {
    "thinker_config": {
        "audio_start_token_id": 5,
        "audio_end_token_id": 6,
        "audio_token_id": 7,
        "audio_config": {
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "encoder_layers": 2,
            "downsample_hidden_size": 8,
            "num_mel_bins": 128,
            "max_source_positions": 128,
            "n_window": 50,
            "n_window_infer": 800,
            "conv_chunksize": 16,
            "output_dim": 32,
            "activation_function": "gelu",
        },
        "text_config": {
            "hidden_size": 32,
            "head_dim": 8,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "attention_bias": False,
            "pad_token_id": 0,
        },
    },
}

ALIGNER = {
    **ASR,
    "timestamp_token_id": 60,
    "timestamp_segment_time": 80,
    "thinker_config": {**ASR["thinker_config"], "classify_num": 8},
}


def audio_inputs(sequence_length=48):
    mask = torch.ones(1, sequence_length, dtype=torch.long)
    features = torch.randn(1, 128, sequence_length)
    return features, mask


def inputs_with_audio(vocab_repeat):
    ids = [5] + [7] * vocab_repeat + [6, 8, 9]
    return torch.tensor([ids], dtype=torch.long)


class TestQwen3ASR:
    def test_feature_length_reduction(self):
        assert feat_extract_lengths(torch.tensor([100])).item() == 13
        assert feat_extract_lengths(torch.tensor([150])).item() == 20

    def test_forward_shapes(self):
        model = Qwen3ASR(Config(**ASR)).eval()
        features, mask = audio_inputs()
        audio_length = int(feat_extract_lengths(mask.sum(-1))[0])
        input_ids = inputs_with_audio(audio_length)
        with torch.no_grad():
            logits = model(input_ids, features, mask)
        assert logits.shape == (1, input_ids.shape[1], 64)

    def test_generate_returns_tokens(self):
        model = Qwen3ASR(Config(**ASR)).eval()
        features, mask = audio_inputs()
        audio_length = int(feat_extract_lengths(mask.sum(-1))[0])
        input_ids = inputs_with_audio(audio_length)
        tokens = model.generate(input_ids, features, mask, max_new_tokens=3)
        assert len(tokens) <= 3
        assert all(0 <= token < 64 for token in tokens)

    def test_forced_aligner_output_length(self):
        model = Qwen3ForcedAligner(Config(**ALIGNER)).eval()
        features, mask = audio_inputs()
        audio_length = int(feat_extract_lengths(mask.sum(-1))[0])
        words = [10, 11, 12]
        ids = [5] + [7] * audio_length + [6]
        for word in words:
            ids += [word, 60, 60]
        input_ids = torch.tensor([ids], dtype=torch.long)
        timestamps = model.align(input_ids, features, mask)
        assert len(timestamps) == len(words) * 2

    def test_pretrained_asr_and_aligner(self, model_path):
        if model_path is None:
            pytest.skip("--model-path required for the end-to-end ASR check")
        from tokenizers import Tokenizer
        from transformers import WhisperFeatureExtractor

        extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        audio = np.zeros(16000, dtype=np.float32)
        processed = extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        features = processed["input_features"]
        mask = processed["attention_mask"]
        audio_length = int(feat_extract_lengths(mask.sum(-1))[0])

        tokenizer = Tokenizer.from_file(f"{model_path}/tokenizer.json")
        audio_tokens = audio_length
        prompt = (
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
            + "<|audio_start|>"
            + "<|audio_pad|>" * audio_tokens
            + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)

        model = Qwen3ASR.from_pretrained(model_path).eval()
        with torch.no_grad():
            logits = model(input_ids, features, mask)
        assert logits.shape[:2] == input_ids.shape

        tokens = model.generate(input_ids, features, mask, max_new_tokens=8)
        assert isinstance(tokens, list)
