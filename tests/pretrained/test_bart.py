import json
from pathlib import Path

import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration as HFBart

from nnm.models.bart import BartForConditionalGeneration

pytestmark = pytest.mark.pretrained


def encode(word, config):
    g2i = {c: i for i, c in enumerate(config.grapheme_chars) if i > 3}
    return torch.tensor([[1] + [g2i[c] for c in word if c in g2i] + [2]])


def decode(ids, config):
    i2p = {i: c for i, c in enumerate(config.phoneme_chars) if i > 3}
    return "".join(i2p.get(int(t), "") for t in ids[0])


def test_pretrained_loads(require_model_path, weight_limit):
    weight_limit(require_model_path)
    model = BartForConditionalGeneration.from_pretrained(require_model_path).eval()
    ids = encode("hello", model.config)
    with torch.no_grad():
        generated = model.generate(ids, max_length=64)
    assert decode(generated, model.config)


def test_matches_upstream(require_model_path, weight_limit):
    weight_limit(require_model_path)
    config_path = Path(require_model_path) / "config.json"
    config = BartConfig(**json.loads(config_path.read_text()))
    model = BartForConditionalGeneration.from_pretrained(require_model_path).eval()
    reference = HFBart(config).eval()
    reference.load_state_dict(model.state_dict(), strict=False)

    for word in ["hello", "world", "kitten", "neural", "speech"]:
        ids = encode(word, model.config)
        with torch.no_grad():
            expected = reference.generate(input_ids=ids, max_length=64, num_beams=1, do_sample=False)
            actual = model.generate(ids, max_length=64)
        assert decode(actual, model.config) == decode(expected, model.config), word
