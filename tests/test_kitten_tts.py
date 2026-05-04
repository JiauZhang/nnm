import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from conippets.config import Config

from nnm.models.kitten_tts import KittenTTS
from nnm.models.albert import AlbertEncoder
from nnm.models.kitten_tts.text_encoder import TextEncoder
from nnm.models.kitten_tts.predictor import Predictor
from nnm.models.kitten_tts.decoder import KittenDecoder
from nnm.models.kitten_tts.generator import KittenGenerator
from nnm.models.kitten_tts.text import text_to_tokens


@pytest.fixture(scope="module")
def config(request):
    model_path = request.config.getoption("--model-path")
    assert model_path is not None, "--model-path required"
    config_path = os.path.join(model_path, 'config.json')
    return Config.from_json(config_path)


@pytest.fixture(scope="module")
def loaded_model(request):
    path = request.config.getoption("--model-path")
    assert path is not None, "--model-path required for integration tests"
    model = KittenTTS.from_pretrained(path)
    model.eval()
    model.generator.m_source.sin_gen.phase_jitter_scale = 0.0
    return model


class TestAlbert:
    @pytest.mark.parametrize("batch,seq_len", [(1, 10), (2, 20)])
    @torch.no_grad()
    def test_albert_output_shape(self, config, batch, seq_len):
        albert = AlbertEncoder(config.albert)
        input_ids = torch.randint(0, config.albert.vocab_size, (batch, seq_len))
        out = albert(input_ids)
        assert out.shape == (batch, seq_len, config.albert.embed_dim)

    @torch.no_grad()
    def test_albert_shared_layer(self, config):
        albert = AlbertEncoder(config.albert)
        input_ids = torch.randint(0, config.albert.vocab_size, (1, 10))
        out1 = albert(input_ids)
        out2 = albert(input_ids)
        torch.testing.assert_close(out1, out2)


class TestTextEncoder:
    @torch.no_grad()
    def test_text_encoder_output_shape(self, config):
        te = TextEncoder(config)
        input_ids = torch.randint(0, config.albert.vocab_size, (1, 10))
        style = torch.randn(1, 256)
        albert_out = torch.randn(1, 10, config.albert.embed_dim)
        lstm_feat, cnn_feat = te(input_ids, style, bert_output=albert_out)
        assert lstm_feat.shape[0] == 1
        assert lstm_feat.shape[2] == 10
        assert cnn_feat.shape[0] == 1
        assert cnn_feat.shape[2] == 10


class TestPredictor:
    @torch.no_grad()
    def test_predictor_output(self, config):
        pred = Predictor(config)
        text = torch.randn(10, 1, 256)
        style = torch.randn(1, 256)
        duration, expanded, shared, f0, n_amp = pred(text, style, speed=1.0)
        assert duration.ndim == 1
        assert len(duration) > 0
        assert f0.shape[0] == 1
        assert n_amp.shape[0] == 1
        assert (duration >= config.predictor.dur_min).all()
        assert (duration <= config.predictor.dur_max).all()

    @torch.no_grad()
    def test_predictor_speed_effect(self, config):
        pred = Predictor(config)
        text = torch.randn(10, 1, 256)
        style = torch.randn(1, 256)
        dur1, _, _, _, _ = pred(text, style, speed=1.0)
        dur2, _, _, _, _ = pred(text, style, speed=2.0)
        assert (dur2 <= dur1).all()


class TestDecoder:
    @torch.no_grad()
    def test_decoder_output_shape(self, config):
        decoder = KittenDecoder(config)
        text = torch.randn(1, 128, 25)
        f0 = torch.randn(1, 1, 50)
        n_amp = torch.randn(1, 1, 50)
        style = torch.randn(1, 256)
        mel = decoder(text, f0, n_amp, style)
        assert mel.shape[0] == 1
        assert mel.shape[1] == config.decoder.decode_out_channels


class TestGenerator:
    @torch.no_grad()
    def test_generator_output_shape(self, config):
        gen = KittenGenerator(config)
        mel = torch.randn(1, 256, 100)
        style = torch.randn(1, config.generator.style_dim)
        f0 = torch.randn(1, 1, 100)
        concat = gen.m_source(f0)
        assert concat.shape[1] == 22
        waveform = gen(mel, style, harmonic_features=concat)
        assert waveform.ndim == 1

    @torch.no_grad()
    def test_m_source_output_shape(self, config):
        gen = KittenGenerator(config)
        f0 = torch.randn(1, 1, 100)
        stft = gen.m_source(f0)
        assert stft.shape[0] == 1
        assert stft.shape[1] == 22


class TestModelLoading:
    def test_from_pretrained_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            KittenTTS.from_pretrained(str(tmp_path))

    def test_voice_lookup(self, loaded_model):
        arr = loaded_model.get_voice("Bruno")
        assert arr.ndim == 2

    def test_unknown_voice_raises(self, loaded_model):
        with pytest.raises(KeyError):
            loaded_model.get_voice("NonExistent")

    def test_speed_priors(self, loaded_model):
        speed = loaded_model.get_speed("Bruno")
        assert speed > 0

    def test_speed_override(self, loaded_model):
        speed = loaded_model.get_speed("Bruno", speed=1.5)
        assert speed == 1.5


class TestSynthesis:
    @torch.no_grad()
    def test_synthesize_output(self, loaded_model):
        result = loaded_model.synthesize("Hello world", voice="Bruno")
        assert "waveform" in result
        assert "duration" in result
        assert result["sample_rate"] == 24000
        assert result["waveform"].ndim == 1
        assert len(result["waveform"]) > 0

    @torch.no_grad()
    def test_synthesize_different_voices(self, loaded_model):
        result1 = loaded_model.synthesize("Hello", voice="Bruno")
        result2 = loaded_model.synthesize("Hello", voice="Bella")
        assert len(result1["waveform"]) > 0
        assert len(result2["waveform"]) > 0

    @torch.no_grad()
    def test_synthesize_deterministic_with_seed(self, loaded_model):
        torch.manual_seed(42)
        np.random.seed(42)
        result1 = loaded_model.synthesize("Test", voice="Bruno")

        torch.manual_seed(42)
        np.random.seed(42)
        result2 = loaded_model.synthesize("Test", voice="Bruno")

        np.testing.assert_array_almost_equal(result1["waveform"], result2["waveform"])

    @torch.no_grad()
    def test_synthesize_speed_affects_duration(self, loaded_model):
        result1 = loaded_model.synthesize("Hello world", voice="Bruno", speed=1.0)
        result2 = loaded_model.synthesize("Hello world", voice="Bruno", speed=2.0)
        assert len(result1["waveform"]) >= len(result2["waveform"])


class TestRegression:
    @torch.no_grad()
    def test_from_pretrained_vs_direct_init(self, loaded_model):
        path = loaded_model.model_path
        voices_path = os.path.join(os.path.dirname(path), 'voices.npz')

        torch.manual_seed(42)
        np.random.seed(42)
        result1 = loaded_model.synthesize("Hello world", voice="Bruno")

        torch.manual_seed(42)
        np.random.seed(42)
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        config = Config.from_json(config_path)
        model2 = KittenTTS(config, model_path=path, voices_path=voices_path)
        model2.eval()
        model2.generator.m_source.sin_gen.phase_jitter_scale = 0.0
        result2 = model2.synthesize("Hello world", voice="Bruno")

        assert len(result1["waveform"]) == len(result2["waveform"])
        assert result1["sample_rate"] == result2["sample_rate"]
        assert result1["voice"] == result2["voice"]
        np.testing.assert_array_almost_equal(result1["duration"], result2["duration"])

    @torch.no_grad()
    def test_long_text_synthesis(self, loaded_model):
        text = "The quick brown fox jumps over the lazy dog. " * 3
        result = loaded_model.synthesize(text, voice="Bruno")
        assert len(result["waveform"]) > 0
        assert result["duration"] is not None


class TestTextProcessing:
    def test_text_to_tokens_basic(self):
        tokens = text_to_tokens("Hello world")
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_text_to_tokens_empty(self):
        with pytest.raises(IndexError):
            text_to_tokens("")

    def test_text_to_tokens_special_chars(self):
        tokens = text_to_tokens("Hello, world! 123")
        assert len(tokens) > 0
