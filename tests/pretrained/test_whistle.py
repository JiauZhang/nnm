import pytest
import torch

from nnm.models.whistle.model import LANGUAGES, Whistle

pytestmark = pytest.mark.pretrained


@pytest.fixture(scope="module")
def loaded_model(load_pretrained):
    return load_pretrained(Whistle)


def noise(seconds=1.0, rate=16000):
    return torch.randn(int(seconds * rate)) * 0.1


class TestWhistleTokenizer:
    @pytest.mark.parametrize("text", ["Turn off the kitchen lights.", "Schalte das Licht in der Küche aus."])
    def test_roundtrip(self, loaded_model, text):
        tokenizer = loaded_model.tokenizer
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_language_markers(self, loaded_model):
        tokenizer = loaded_model.tokenizer
        ids = tokenizer.encode("<|de|>hello")
        assert "de" == [marker[2:-2] for marker in tokenizer.markers if tokenizer.piece_to_id[marker] in ids][0]


class TestWhistleFrontend:
    def test_frame_count(self, loaded_model):
        samples = noise(2.0)
        with torch.no_grad():
            embed = loaded_model.embed(samples)
        assert embed.shape == (samples.numel() // 1280, loaded_model.config.d_model)


class TestWhistleContract:
    def test_silence_is_an_empty_transcript(self, loaded_model):
        assert loaded_model.transcribe(torch.zeros(16000)) == {"text": "", "language": "", "ttft_ms": 0.0, "decode_tps": 0.0}

    def test_steady_tone_is_an_empty_transcript(self, loaded_model):
        tone = torch.sin(2 * torch.pi * 440 * torch.arange(16000) / 16000)
        assert loaded_model.transcribe(tone)["text"] == ""

    def test_reports_the_published_keys(self, loaded_model):
        result = loaded_model.transcribe(noise(2.0))
        assert set(result) == {"text", "language", "ttft_ms", "decode_tps"}
        assert result["language"] in LANGUAGES + ("",)
