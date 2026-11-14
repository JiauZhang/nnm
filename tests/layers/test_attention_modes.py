import torch

import nnm.layers.attention as attention_layer
from nnm.cache import KVCache, SlidingWindowKVCache


class RecordingSDPA:
    def __init__(self):
        self.calls = []

    def __call__(self, query, key, value, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros_like(query)


def run(monkeypatch, query, key=None, value=None, **kwargs):
    recorder = RecordingSDPA()
    monkeypatch.setattr(attention_layer, "scaled_dot_product_attention", recorder)
    query = query.transpose(1, 2)
    key = query if key is None else key.transpose(1, 2)
    value = query if value is None else value.transpose(1, 2)
    attention_layer.causal_attention(query, key, value, **kwargs)
    return recorder.calls[-1]


def test_training_causal_uses_no_mask(monkeypatch):
    query = torch.randn(1, 4, 8, 16)

    call = run(monkeypatch, query)

    assert call["attn_mask"] is None
    assert call["is_causal"] is True


def test_decode_with_cache_uses_no_mask(monkeypatch):
    cache = KVCache(capacity=16)
    cache.update(torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16))
    query = torch.randn(1, 1, 4, 16)

    call = run(monkeypatch, query, cache=cache)

    assert cache.get_seq_length() == 6
    assert call["attn_mask"] is None
    assert call["is_causal"] is False


def test_sliding_decode_uses_no_mask(monkeypatch):
    cache = SlidingWindowKVCache(window_size=4)
    cache.update(torch.randn(1, 4, 4, 16), torch.randn(1, 4, 4, 16))
    query = torch.randn(1, 1, 4, 16)

    call = run(monkeypatch, query, cache=cache, sliding_window=4)

    assert call["attn_mask"] is None
    assert call["is_causal"] is False


def test_chunked_prefill_builds_a_mask(monkeypatch):
    cache = KVCache(capacity=16)
    cache.update(torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16))
    query = torch.randn(1, 2, 4, 16)

    call = run(monkeypatch, query, cache=cache)

    assert call["attn_mask"] is not None
    assert call["is_causal"] is False


def test_padding_builds_a_mask(monkeypatch):
    query = torch.randn(1, 4, 8, 16)
    padding = torch.ones(1, 8)

    call = run(monkeypatch, query, attn_mask=padding)

    assert call["attn_mask"] is not None
    assert call["is_causal"] is False
