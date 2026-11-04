import pytest
import torch

from conippets.config import Config
from nnm.models.qwen3_tts import Qwen3TTS, Qwen3TTSCodec

TALKER = {
    "tts_bos_token_id": 5,
    "tts_eos_token_id": 6,
    "tts_pad_token_id": 7,
    "talker_config": {
        "hidden_size": 32,
        "head_dim": 8,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "text_vocab_size": 100,
        "text_hidden_size": 16,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "num_code_groups": 4,
        "codec_eos_token_id": 26,
        "codec_pad_id": 24,
        "codec_bos_id": 25,
        "codec_nothink_id": 20,
        "codec_think_id": 21,
        "codec_think_bos_id": 22,
        "codec_think_eos_id": 23,
        "spk_id": {"ryan": 10},
        "codec_language_id": {"english": 11},
        "spk_is_dialect": {"ryan": False},
        "code_predictor_config": {
            "hidden_size": 32,
            "head_dim": 8,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "num_code_groups": 4,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "attention_bias": False,
        },
    },
}

CODEC = {
    "decoder_config": {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "sliding_window": 8,
        "max_position_embeddings": 64,
        "latent_dim": 32,
        "codebook_dim": 16,
        "codebook_size": 8,
        "decoder_dim": 32,
        "num_quantizers": 4,
        "num_semantic_quantizers": 1,
        "upsample_rates": [2, 2, 2, 2],
        "upsampling_ratios": [2, 2],
        "layer_scale_initial_scale": 0.01,
        "attention_bias": False,
    }
}


class TestQwen3TTS:
    def test_prompt_embedding_layout(self):
        model = Qwen3TTS(Config(**TALKER)).eval()
        input_ids = torch.arange(30, 42).unsqueeze(0)
        with torch.no_grad():
            talker_embed, trailing, pad = model.embed_prompt(input_ids, "Ryan", "English")
        assert talker_embed.shape == (1, 15, 32)
        assert trailing.shape == (1, 1, 32)
        assert pad.shape == (1, 1, 32)

    def test_generate_codes_returns_a_full_frame(self):
        model = Qwen3TTS(Config(**TALKER)).eval()
        input_ids = torch.arange(30, 42).unsqueeze(0)
        with torch.no_grad():
            codes = model.generate_codes(input_ids, "Ryan", "English", max_new_tokens=3)
        assert codes.shape[1] == 4
        assert len(codes) <= 3

    def test_pretrained_talker_and_codec(self, model_path):
        if model_path is None:
            pytest.skip("--model-path required for the end-to-end TTS check")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        text = "<|im_start|>assistant\nHello there.<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

        talker = Qwen3TTS.from_pretrained(model_path).eval()
        codes = talker.generate_codes(input_ids, "Ryan", "English", max_new_tokens=4)
        assert codes.shape[1] == 16
        assert len(codes) > 0

        codec = Qwen3TTSCodec.from_pretrained(model_path).eval()
        wav = codec.decode(codes)
        assert wav.shape[:2] == (1, 1)
        assert wav.shape[-1] == len(codes) * 1920
