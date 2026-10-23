import pytest
import torch
from transformers import GPT2LMHeadModel

from nnm.models.gpt2 import GPT2LM


@pytest.fixture(scope="module")
def model(model_path):
    assert model_path is not None, "--model-path required"
    return GPT2LM.from_pretrained(model_path, use_cache=True).eval()


@pytest.fixture(scope="module")
def reference(reference_path):
    if reference_path is None:
        pytest.skip("--reference-path required for the transformers comparison")
    return GPT2LMHeadModel.from_pretrained(reference_path).eval()


@pytest.fixture(autouse=True)
def empty_cache(model):
    model.clear_cache()


def make_inputs(batch=2, length=16):
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 50257, (batch, length), generator=generator)
    attention_mask = torch.ones(batch, length, dtype=torch.int64)
    attention_mask[1, 11:] = 0
    input_ids[1, 11:] = 0
    return input_ids, attention_mask


def test_param_count(model):
    assert sum(p.numel() for p in model.parameters()) == 124439808


def test_logits_match_transformers(model, reference):
    input_ids, attention_mask = make_inputs()
    with torch.no_grad():
        expected = reference(input_ids, attention_mask=attention_mask).logits
        got = model(input_ids, attention_mask)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4)


def test_incremental_decoding_matches_the_full_pass(model):
    input_ids, _ = make_inputs()
    with torch.no_grad():
        full = model(input_ids)
        model.clear_cache()
        chunks = [model(input_ids[:, :8]), model(input_ids[:, 8:12]), model(input_ids[:, 12:])]

    assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-4)


def test_cache_tracks_the_context(model):
    input_ids, _ = make_inputs(length=6)
    with torch.no_grad():
        model(input_ids[:, :4])
        assert model.backbone.caches[0].kv_len == 4
        model(input_ids[:, 4:])

    assert model.backbone.caches[0].kv_len == 6
