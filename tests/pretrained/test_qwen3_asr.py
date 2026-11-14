import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor

from nnm.models.qwen3_asr import Qwen3ASR
from nnm.models.qwen3_asr.audio import feat_extract_lengths

pytestmark = pytest.mark.pretrained


def test_pretrained_asr_and_aligner(require_model_path):
    extractor = WhisperFeatureExtractor.from_pretrained(require_model_path)
    audio = np.zeros(16000, dtype=np.float32)
    processed = extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    features = processed["input_features"]
    mask = processed["attention_mask"]
    audio_length = int(feat_extract_lengths(mask.sum(-1))[0])

    tokenizer = Tokenizer.from_file(f"{require_model_path}/tokenizer.json")
    prompt = (
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
        + "<|audio_start|>"
        + "<|audio_pad|>" * audio_length
        + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    )
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)

    model = Qwen3ASR.from_pretrained(require_model_path).eval()
    with torch.no_grad():
        logits = model(input_ids, features, mask)
    assert logits.shape[:2] == input_ids.shape

    tokens = model.generate(input_ids, features, mask, max_new_tokens=8)
    assert isinstance(tokens, list)
