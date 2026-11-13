import json
import os

import pytest
import torch
from conippets.config import Config

from nnm.models.bart import BartForConditionalGeneration

REPO = "graphemes_to_phonemes_en_us"


def _encode(word, config):
    g2i = {c: i for i, c in enumerate(config.grapheme_chars) if i > 3}
    return torch.tensor([[1] + [g2i[c] for c in word if c in g2i] + [2]])


def _decode(ids, config):
    i2p = {i: c for i, c in enumerate(config.phoneme_chars) if i > 3}
    return "".join(i2p.get(int(t), "") for t in ids[0])


def test_architecture(weight_config):
    config = weight_config(REPO)
    model = BartForConditionalGeneration(config).eval()
    input_ids = torch.randint(4, config.vocab_size, (1, 8))
    decoder_input_ids = torch.randint(4, config.vocab_size, (1, 5))
    with torch.no_grad():
        logits = model(input_ids, decoder_input_ids)
        generated = model.generate(input_ids, max_length=16)
    assert logits.shape == (1, 5, config.vocab_size)
    assert generated.shape[0] == 1
    assert torch.isfinite(logits).all()


def test_pretrained_loads(model_path, weight_limit):
    if model_path is None:
        pytest.skip("--model-path required")
    weight_limit(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).eval()
    ids = _encode("hello", model.config)
    with torch.no_grad():
        generated = model.generate(ids, max_length=64)
    assert _decode(generated, model.config)


def test_matches_upstream(model_path, weight_limit):
    if model_path is None:
        pytest.skip("--model-path required")
    weight_limit(model_path)
    from transformers import BartConfig, BartForConditionalGeneration as HFBart  # noqa: PLC0415

    config = Config.from_json(os.path.join(model_path, "config.json"))
    model = BartForConditionalGeneration.from_pretrained(model_path).eval()
    reference = HFBart(BartConfig(**json.load(open(os.path.join(model_path, "config.json"))))).eval()
    reference.load_state_dict(model.state_dict(), strict=False)

    for word in ["hello", "world", "kitten", "neural", "speech"]:
        ids = _encode(word, config)
        with torch.no_grad():
            ref = reference.generate(input_ids=ids, max_length=64, num_beams=1, do_sample=False)
            out = model.generate(ids, max_length=64)
        assert _decode(out, config) == _decode(ref, config), word
