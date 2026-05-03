import torch
import pytest
from nnm.cache import KVCache, SlidingWindowKVCache


def test_kv_cache_basic():
    cache = KVCache()
    assert cache.is_empty()
    assert cache.kv_len == 0

    k1 = torch.randn(2, 4, 3, 16)
    v1 = torch.randn(2, 4, 3, 16)
    k_out, v_out = cache.update(k1, v1)

    assert not cache.is_empty()
    assert cache.kv_len == 3
    assert torch.equal(k_out, k1)
    assert torch.equal(v_out, v1)

    k2 = torch.randn(2, 4, 5, 16)
    v2 = torch.randn(2, 4, 5, 16)
    k_out, v_out = cache.update(k2, v2)

    assert cache.kv_len == 8
    assert torch.equal(k_out, torch.cat([k1, k2], dim=-2))
    assert torch.equal(v_out, torch.cat([v1, v2], dim=-2))

    cache.clear()
    assert cache.is_empty()
    assert cache.kv_len == 0


@pytest.mark.parametrize(
    "seq_lens,expected_fn",
    [
        ([3, 5], lambda ks: torch.cat(ks, dim=-2)),
        ([6, 5], lambda ks: torch.cat(ks, dim=-2)[:, :, -10:, :]),
        ([10, 3], lambda ks: torch.cat(ks, dim=-2)[:, :, -10:, :]),
        ([10, 15], lambda ks: ks[1][:, :, -10:, :]),
        ([15], lambda ks: ks[0][:, :, -10:, :]),
    ],
)
def test_sliding_window_cache(seq_lens, expected_fn):
    cache = SlidingWindowKVCache(window_size=10)
    k_list = [torch.randn(2, 4, s, 16) for s in seq_lens]
    v_list = [torch.randn(2, 4, s, 16) for s in seq_lens]

    for k, v in zip(k_list, v_list):
        k_out, v_out = cache.update(k, v)

    assert cache.kv_len == min(sum(seq_lens), 10)
    assert torch.equal(k_out, expected_fn(k_list))
    assert torch.equal(v_out, expected_fn(v_list))


def test_sliding_window_cache_clear():
    cache = SlidingWindowKVCache(window_size=10)
    k = torch.randn(2, 4, 5, 16)
    v = torch.randn(2, 4, 5, 16)
    cache.update(k, v)

    cache.clear()
    assert cache.is_empty()
    assert cache.kv_len == 0


def test_sliding_window_cache_multiple_updates_at_limit():
    cache = SlidingWindowKVCache(window_size=10)
    k1 = torch.randn(2, 4, 10, 16)
    v1 = torch.randn(2, 4, 10, 16)
    cache.update(k1, v1)

    for _ in range(5):
        k = torch.randn(2, 4, 2, 16)
        v = torch.randn(2, 4, 2, 16)
        k_out, v_out = cache.update(k, v)
        assert cache.kv_len == 10
        assert torch.equal(k_out, torch.cat([cache._k_cache[:, :, :-2, :], k], dim=-2))
