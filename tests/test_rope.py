import pytest
import torch
from transformers.models.qwen2 import configuration_qwen2 as cfg
from transformers.models.qwen2 import modeling_qwen2 as qwen2

from nnm.layers.rope import RoPE, RoPE2D, apply_rope, rotary_inv_freq


@pytest.mark.parametrize(
    "seq_len, embed_dim, base, start",
    [
        (256, 32, 6666, 0),
        (128, 64, 2345, 0),
        (512, 64, 10000, 32),
        (999, 64, 10000, 66),
    ],
)
def test_rope_matches_qwen2(seq_len, embed_dim, base, start):
    positions = torch.arange(start, start + seq_len)
    rope = RoPE(rotary_inv_freq(embed_dim, base))
    nnm_cos, nnm_sin = rope(positions)

    config = cfg.Qwen2Config(head_dim=embed_dim, rope_theta=base, max_position_embeddings=start + seq_len)
    hf_rope = qwen2.Qwen2RotaryEmbedding(config=config)
    hf_cos, hf_sin = hf_rope(torch.empty(1, seq_len, embed_dim), positions.unsqueeze(0))

    torch.testing.assert_close(nnm_cos.unsqueeze(0), hf_cos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(nnm_sin.unsqueeze(0), hf_sin, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("head_dim", [16, 32])
def test_partial_rope_keeps_the_tail_fixed(head_dim):
    rope = RoPE(rotary_inv_freq(head_dim, 10000, rotary_dim=head_dim // 2))
    cos, sin = rope(torch.arange(4))
    cos, sin = cos[0], sin[0]
    assert torch.allclose(cos[head_dim // 2 :], torch.ones_like(cos[head_dim // 2 :]))
    assert torch.allclose(sin[head_dim // 2 :], torch.zeros_like(sin[head_dim // 2 :]))
    x = torch.randn(4, head_dim)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(rotated[:, head_dim // 2 :], x[:, head_dim // 2 :])


def test_rope2d_concatenates_axes():
    inv_freq = rotary_inv_freq(8, 10000)
    rope = RoPE2D([inv_freq, inv_freq])
    positions = torch.tensor([[1, 2], [3, 4]])
    cos, sin = rope(positions)
    assert cos.shape == (2, 16)

    angles = torch.cat([positions[:, 0:1].float() * inv_freq, positions[:, 1:2].float() * inv_freq], dim=-1)
    expected_cos = torch.cat([angles, angles], dim=-1).cos()
    torch.testing.assert_close(cos, expected_cos, atol=1e-6, rtol=1e-6)
