from functools import lru_cache

import pytest
import torch

from nnm.models.kronos import Kronos, KronosTokenizer

BACKBONES = {
    'Kronos-mini': (256, 4, 4, 4108032),
    'Kronos-small': (512, 8, 8, 24741376),
    'Kronos-base': (832, 12, 16, 102310592),
}

TOKENIZERS = {'Kronos-Tokenizer-base': 4, 'Kronos-Tokenizer-2k': 5}


@lru_cache(maxsize=None)
def load_tokenizer(name):
    return KronosTokenizer.from_pretrained(f'JiauZhang/{name}').eval()


@lru_cache(maxsize=None)
def load_model(name):
    return Kronos.from_pretrained(f'JiauZhang/{name}').eval()


def make_series(batch=2, length=16, features=6):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, length, features, generator=generator)


def make_tokens(batch=2, length=16, bits=10):
    generator = torch.Generator().manual_seed(0)
    s1 = torch.randint(0, 2**bits, (batch, length), generator=generator)
    s2 = torch.randint(0, 2**bits, (batch, length), generator=generator)
    stamp = torch.stack([torch.randint(0, 7, (batch, length), generator=generator) for _ in range(5)], -1).float()
    return s1, s2, stamp


@pytest.mark.parametrize('name', list(TOKENIZERS))
class TestKronosTokenizer:
    def test_config_drives_the_architecture(self, name):
        tokenizer = load_tokenizer(name)
        assert tokenizer.config.d_model == 256
        assert tokenizer.config.n_enc_layers == 4
        assert tokenizer.config.group_size == TOKENIZERS[name]
        assert sum(p.numel() for p in tokenizer.parameters()) == 3955226

    def test_encode_returns_two_ten_bit_streams(self, name):
        tokenizer = load_tokenizer(name)
        series = make_series()
        with torch.no_grad():
            pre, post = tokenizer.encode(series, half=True)
            full = tokenizer.encode(series, half=False)

        assert pre.shape == post.shape == full.shape == series.shape[:2]
        assert pre.min() >= 0 and pre.max() < 2**10
        assert post.min() >= 0 and post.max() < 2**10
        assert torch.equal(full, pre + post * 2**10)

    def test_decode_matches_the_input_shape(self, name):
        tokenizer = load_tokenizer(name)
        series = make_series()
        with torch.no_grad():
            indices = tokenizer.encode(series, half=False)
            restored = tokenizer.decode(indices)

        assert restored.shape == series.shape
        with torch.no_grad():
            assert torch.equal(tokenizer.decode(indices), tokenizer.decode(indices))

    def test_half_and_full_decode_agree(self, name):
        tokenizer = load_tokenizer(name)
        series = make_series()
        with torch.no_grad():
            pre, post = tokenizer.encode(series, half=True)
            full = tokenizer.encode(series, half=False)

        assert torch.equal(tokenizer.decode(full), tokenizer.decode((pre, post), half=True))


@pytest.mark.parametrize('name', list(BACKBONES))
class TestKronos:
    def test_config_drives_the_architecture(self, name):
        model = load_model(name)
        d_model, n_layers, n_heads, parameters = BACKBONES[name]
        assert model.config.d_model == d_model
        assert model.config.n_layers == n_layers
        assert model.config.n_heads == n_heads
        assert sum(p.numel() for p in model.parameters()) == parameters

    def test_forward_returns_both_heads(self, name):
        model = load_model(name)
        s1, s2, stamp = make_tokens()
        with torch.no_grad():
            s1_logits, s2_logits = model(s1, s2, stamp)

        assert s1_logits.shape == (2, 16, 1024)
        assert s2_logits.shape == (2, 16, 1024)

    def test_decode_s2_reproduces_the_conditional_head(self, name):
        model = load_model(name)
        s1, s2, stamp = make_tokens()
        with torch.no_grad():
            s1_logits, s2_logits = model(s1, s2, stamp)
            head_logits, context = model.decode_s1(s1, s2, stamp)

        assert torch.equal(s1_logits, head_logits)
        assert torch.allclose(model.decode_s2(context, s1), s2_logits, atol=1e-6)

    def test_timestamps_shift_the_logits(self, name):
        model = load_model(name)
        s1, s2, stamp = make_tokens()
        with torch.no_grad():
            without = model(s1, s2)[0]
            with_stamp = model(s1, s2, stamp)[0]

        assert without.shape == with_stamp.shape
        assert not torch.allclose(without, with_stamp)

    def test_padding_mask_only_touches_the_masked_tail(self, name):
        model = load_model(name)
        s1, s2, stamp = make_tokens()
        padding = torch.ones(2, 16, dtype=torch.bool)
        padding[1, 10:] = False

        with torch.no_grad():
            full = model(s1, s2, stamp)[0]
            masked = model(s1, s2, stamp, padding_mask=padding)[0]

        assert torch.allclose(full[:, :10], masked[:, :10], atol=1e-6)
        assert not torch.allclose(full[1, 10:], masked[1, 10:])
