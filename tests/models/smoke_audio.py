import torch

from nnm.models.qwen3_asr.audio import feat_extract_lengths
from tests.models.smoke_registry import SmokeCase, make


def check_audio8_asr(model, output):
    assert tuple(output.shape) == (1, 6, 100)
    input_ids, _, _ = model_inputs()
    text_only = model(input_ids)
    assert not torch.allclose(output, text_only)


def check_audio8_tts(model, output):
    logits, hidden = output
    assert tuple(logits.shape) == (2, 5, 33)
    assert tuple(hidden.shape) == (2, 5, 32)


def model_inputs(audio_token_id=5, num_mel_bins=8, frames=16):
    input_ids = torch.tensor([[0, audio_token_id, audio_token_id, audio_token_id, 6, 7]])
    input_features = torch.randn(1, num_mel_bins, frames)
    feature_lens = torch.tensor([frames])
    return input_ids, input_features, feature_lens


def tts_inputs():
    semantic = torch.randint(100, 111, (2, 5))
    codebooks = torch.randint(0, 32, (2, 3, 5))
    input_ids = torch.cat([semantic[:, None], codebooks], dim=1)
    return input_ids, torch.ones(2, 5, dtype=torch.long)


def asr_inputs():
    mask = torch.ones(1, 48, dtype=torch.long)
    features = torch.randn(1, 128, 48)
    audio_length = int(feat_extract_lengths(mask.sum(-1))[0])
    input_ids = torch.tensor([[5] + [7] * audio_length + [6, 8, 9]], dtype=torch.long)
    return input_ids, features, mask


def aligner_inputs():
    input_ids, features, mask = asr_inputs()
    audio_length = int(feat_extract_lengths(mask.sum(-1))[0])
    ids = [5] + [7] * audio_length + [6]
    for word in (10, 11, 12):
        ids += [word, 60, 60]
    return torch.tensor([ids], dtype=torch.long), features, mask


def check_qwen3_asr(model, output):
    input_ids, features, mask = asr_inputs()
    assert tuple(output.shape) == (1, input_ids.shape[1], 64)
    tokens = model.generate(input_ids, features, mask, max_new_tokens=3)
    assert len(tokens) <= 3
    assert all(0 <= token < 64 for token in tokens)


def check_qwen3_aligner(model, output):
    assert len(output) == 6


def check_qwen3_tts(model, output):
    talker_embed, trailing, pad = model.embed_prompt(torch.arange(30, 42).unsqueeze(0), "Ryan", "English")
    assert tuple(talker_embed.shape) == (1, 15, 32)
    assert tuple(trailing.shape) == (1, 1, 32)
    assert tuple(pad.shape) == (1, 1, 32)
    codes = model.generate_codes(torch.arange(30, 42).unsqueeze(0), "Ryan", "English", max_new_tokens=3)
    assert codes.shape[1] == 4
    assert len(codes) <= 3


ASR_CONFIG = {
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
}

CASES = [
    SmokeCase(
        "audio8-asr",
        "mlp-tower",
        make(
            "nnm.models.audio8_asr",
            "Audio8ASR",
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
        ),
        model_inputs,
        check_audio8_asr,
    ),
    SmokeCase(
        "audio8-tts",
        "dual-ar",
        make(
            "nnm.models.audio8_tts",
            "Audio8TTS",
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
            norm_eps=1e-5,
            hidden_act="silu",
            rope_theta=10000.0,
            rope_base=10000.0,
            pad_token_id=0,
            eos_token_id=127,
            embedding_multiplier=0.1,
            mamba_d_ssm=64,
            mamba_d_head=16,
            mamba_d_state=16,
            mamba_n_heads=4,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_chunk_size=8,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            projectors_bias=False,
            mlp_bias=False,
            n_fast_layer=2,
            fast_dim=32,
            fast_n_head=4,
            fast_n_local_heads=2,
            fast_head_dim=8,
            fast_intermediate_size=64,
            fast_attention_qkv_bias=False,
            fast_attention_o_bias=False,
            codebook_size=32,
            num_codebooks=3,
            semantic_begin_id=100,
            semantic_end_id=110,
        ),
        tts_inputs,
        check_audio8_tts,
    ),
]

ASR_MODEL_CONFIG = {"thinker_config": ASR_CONFIG}

ALIGNER_MODEL_CONFIG = {
    "thinker_config": {**ASR_CONFIG, "classify_num": 8},
    "timestamp_token_id": 60,
    "timestamp_segment_time": 80,
}

TALKER_CONFIG = {
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
}

CASES.extend(
    [
        SmokeCase(
            "qwen3-asr",
            "thinker",
            make("nnm.models.qwen3_asr", "Qwen3ASR", thinker_config=ASR_CONFIG),
            asr_inputs,
            check_qwen3_asr,
        ),
        SmokeCase(
            "qwen3-asr",
            "aligner",
            make("nnm.models.qwen3_asr", "Qwen3ForcedAligner", **ALIGNER_MODEL_CONFIG),
            lambda: (),
            check_qwen3_aligner,
            call=lambda model: model.align(*aligner_inputs()),
        ),
        SmokeCase(
            "qwen3-tts",
            "talker",
            make(
                "nnm.models.qwen3_tts",
                "Qwen3TTS",
                tts_bos_token_id=5,
                tts_eos_token_id=6,
                tts_pad_token_id=7,
                talker_config=TALKER_CONFIG,
            ),
            lambda: (),
            check_qwen3_tts,
            call=lambda model: model.generate_codes(torch.arange(30, 42).unsqueeze(0), "Ryan", "English", max_new_tokens=3),
        ),
    ]
)
