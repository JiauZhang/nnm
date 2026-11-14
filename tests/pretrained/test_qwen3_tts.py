import pytest
from transformers import AutoTokenizer

from nnm.models.qwen3_tts import Qwen3TTS, Qwen3TTSCodec

pytestmark = pytest.mark.pretrained


def test_pretrained_talker_and_codec(require_model_path):
    tokenizer = AutoTokenizer.from_pretrained(require_model_path)
    text = "<|im_start|>assistant\nHello there.<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    talker = Qwen3TTS.from_pretrained(require_model_path).eval()
    codes = talker.generate_codes(input_ids, "Ryan", "English", max_new_tokens=4)
    assert codes.shape[1] == 16
    assert len(codes) > 0

    codec = Qwen3TTSCodec.from_pretrained(require_model_path).eval()
    wav = codec.decode(codes)
    assert wav.shape[:2] == (1, 1)
    assert wav.shape[-1] == len(codes) * 1920
