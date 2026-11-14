import pytest
import torch

from nnm.cache import Cache, KVCache, SlidingWindowKVCache, StateCache


def test_kv_cache_writes_in_place():
    cache = KVCache(capacity=8)
    assert cache.is_empty()
    assert cache.get_seq_length() == 0
    assert cache.get_mask_sizes(2) == (2, 0)

    k1 = torch.randn(2, 4, 3, 16)
    v1 = torch.randn(2, 4, 3, 16)
    k_out, v_out = cache.update(k1, v1)

    assert not cache.is_empty()
    assert cache.get_seq_length() == 3
    assert cache.get_mask_sizes(2) == (5, 0)
    assert torch.equal(k_out, k1)
    assert torch.equal(v_out, v1)

    k2 = torch.randn(2, 4, 5, 16)
    v2 = torch.randn(2, 4, 5, 16)
    k_out, v_out = cache.update(k2, v2)

    assert cache.get_seq_length() == 8
    assert torch.equal(k_out, torch.cat([k1, k2], dim=-2))
    assert torch.equal(v_out, torch.cat([v1, v2], dim=-2))

    cache.clear()
    assert cache.is_empty()
    assert cache.get_seq_length() == 0


def test_kv_cache_reserves_more_capacity():
    cache = KVCache(capacity=2)
    first = torch.randn(2, 4, 2, 16)
    cache.update(first, first)

    second = torch.randn(2, 4, 3, 16)
    cache.update(second, second)

    assert cache.capacity >= 5
    keys, _, length = cache._keys, cache._values, cache.get_seq_length()
    assert length == 5
    assert torch.equal(keys[..., :5, :], torch.cat([first, second], dim=-2))


def test_sliding_window_cache_returns_the_window_context():
    cache = SlidingWindowKVCache(window_size=4)
    chunks = [torch.randn(2, 4, size, 16) for size in (3, 2, 5)]

    returned = [cache.update(chunk, chunk)[0] for chunk in chunks]

    for index, chunk in enumerate(chunks):
        context = torch.cat(chunks[:index], dim=-2) if index else None
        expected = chunk if context is None else torch.cat([context[:, :, -3:, :], chunk], dim=-2)
        assert torch.equal(returned[index], expected)


def test_sliding_window_cache_keeps_the_window():
    cache = SlidingWindowKVCache(window_size=10)

    k1 = torch.randn(2, 4, 10, 16)
    cache.update(k1, k1)
    assert cache.get_seq_length() == 10
    assert cache.stored_length == 9

    k2 = torch.randn(2, 4, 15, 16)
    cache.update(k2, k2)
    assert cache.get_seq_length() == 25
    assert cache.stored_length == 9
    assert torch.equal(cache._keys[..., :9, :], torch.cat([k1, k2], dim=-2)[:, :, -9:, :])


def test_sliding_window_decode_is_bounded():
    cache = SlidingWindowKVCache(window_size=4)

    for index in range(10):
        token = torch.full((1, 2, 1, 8), index)
        keys, _ = cache.update(token, token)
        assert keys.shape[-2] <= 4

    assert cache.get_seq_length() == 10
    assert cache.get_mask_sizes(1) == (4, 7)


def test_state_cache():
    cache = StateCache(state_size=4)
    assert cache.is_empty()
    assert cache.get_seq_length() == 0

    state = cache.update(torch.randn(2, 8, 3))
    assert torch.equal(state, cache.state)
    assert state.shape == (2, 8, 4)

    cache.clear()
    assert cache.is_empty()


def test_cache_container_tracks_positions_and_masks():
    cache = Cache.from_layer_types(["conv", "full_attention", "sliding_attention"], capacity=8, window_size=4, state_size=3)
    assert len(cache) == 3
    assert isinstance(cache[0], StateCache)
    assert isinstance(cache[1], KVCache)
    assert isinstance(cache[2], SlidingWindowKVCache)
    assert cache.get_seq_length() == 0
    assert torch.equal(cache.get_positions(2), torch.tensor([0, 1]))

    k = torch.randn(2, 4, 3, 16)
    cache[1].update(k, k)
    cache[2].update(k, k)
    assert cache.get_seq_length() == 3
    assert cache.get_mask_sizes(1) == (4, 0)
    assert torch.equal(cache.get_positions(2), torch.tensor([3, 4]))

    cache.reserve(16)
    assert cache[1].capacity == 16

    cache.clear()
    assert cache.is_empty()


def test_cache_factory_requires_a_window_for_sliding_layers():
    with pytest.raises(ValueError):
        Cache.from_layer_types(["sliding_attention"], capacity=8, window_size=None)
