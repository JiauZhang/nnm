import os

import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import ModernBertConfig, ModernBertModel

from nnm.models.laya.model import Laya


@pytest.fixture(scope="module")
def model(model_path):
    assert model_path is not None, "--model-path required"
    return Laya.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def weights(model_path):
    return load_file(os.path.join(model_path, "model.safetensors"))


def make_inputs(batch=2, length=32):
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 50000, (batch, length), generator=generator)
    attention_mask = torch.ones(batch, length, dtype=torch.int64)
    attention_mask[1, length // 2 :] = 0
    input_ids[1, length // 2 :] = 0
    marker_pos = torch.randint(1, length // 2, (batch, 4), generator=generator)
    marker_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    qtype = torch.tensor([0, 1])
    return input_ids, attention_mask, marker_pos, marker_mask, qtype


def reference_head(weights, hidden_size=1024, head_layers=2):
    head = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(hidden_size, 16, 4 * hidden_size, 0.1, batch_first=True, norm_first=True),
        head_layers,
        enable_nested_tensor=False,
    )
    head.load_state_dict(
        {name.removeprefix("head."): value for name, value in weights.items() if name.startswith("head.")},
        strict=True,
    )
    return head.eval()


def test_encoder_matches_transformers(model, model_path):
    config = ModernBertConfig.from_pretrained(os.path.join(model_path, "encoder"))
    reference = ModernBertModel(config).eval()
    encoder = {
        name.removeprefix("encoder."): value
        for name, value in load_file(os.path.join(model_path, "model.safetensors")).items()
        if name.startswith("encoder.")
    }
    reference.load_state_dict(encoder, strict=True)

    input_ids, attention_mask, *_ = make_inputs()
    with torch.no_grad():
        expected = reference(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        got = model.encoder(input_ids, attention_mask)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4)


def test_decision_head_matches_reference(model, weights):
    input_ids, attention_mask, marker_pos, marker_mask, qtype = make_inputs()
    with torch.no_grad():
        logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        seq = model.encoder(input_ids, attention_mask) + model.type_emb(qtype)[:, None, :]
        for layer in reference_head(weights).layers:
            seq = layer(seq, src_key_padding_mask=~attention_mask.bool())
        expected = model.scorer(torch.gather(seq, 1, marker_pos[:, :, None].expand(-1, -1, seq.size(-1)))).squeeze(-1)
        expected = expected.masked_fill(~marker_mask.bool(), -1e4)

    assert model.temperature.tolist() == weights["temperature"].tolist()
    assert logits.shape == (2, 4)
    assert torch.allclose(logits, expected, atol=1e-4)
    assert torch.all(logits[~marker_mask.bool()] == -1e4)
    assert act.shape == (2, 2)
    assert torch.isfinite(act).all()


def test_action_features_follow_the_probs(model, weights):
    input_ids, attention_mask, marker_pos, marker_mask, qtype = make_inputs()
    with torch.no_grad():
        logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        probs = torch.softmax(logits, -1)
        count = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(probs * probs.clamp(min=1e-9).log()).sum(-1) / count.log()
        top = probs.topk(2, -1).values
        features = torch.stack([top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255.0], dim=-1)
        hidden = model.encoder(input_ids, attention_mask) + model.type_emb(qtype)[:, None, :]
        for layer in reference_head(weights).layers:
            hidden = layer(hidden, src_key_padding_mask=~attention_mask.bool())
        expected = model.act_head(torch.cat([hidden[:, 0], features], dim=-1))

    assert torch.allclose(act, expected, atol=1e-3)
