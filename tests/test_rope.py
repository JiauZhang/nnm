import torch, pytest
from torch import nn
from nnm.layers.rope import RoPE, QwenRoPE
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from transformers.models.qwen2 import configuration_qwen2 as cfg

def llama_precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def llama_reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L132
def llama_apply_rotary_emb(xq, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = llama_reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)

@pytest.mark.parametrize(
    'seq_len, embed_dim, max_seq_len, base, cache, position', [
        (256, 32, 1024, 6666, False, None),
        (128, 64, 1024, 10000, False, None),
        (512, 64, 1024, 10000, True, 32),
        (999, 64, 1024, 10000, True, 66),
    ]
)
def test_rope(seq_len, embed_dim, max_seq_len, base, cache, position):
    rope = RoPE(max_seq_len=max_seq_len, embed_dim=embed_dim, base=base)
    x = torch.randn(1, seq_len, embed_dim)
    y = rope(x)
    llama_freqs_cis = llama_precompute_freqs_cis(embed_dim, max_seq_len, base)[:seq_len]
    llama_y = llama_apply_rotary_emb(x, llama_freqs_cis)
    llama_y = llama_y.flatten(-2)

    assert y.shape == llama_y.shape
    assert torch.abs(llama_y - y).mean() < 1e-5

@pytest.mark.parametrize(
    'seq_len, embed_dim, max_seq_len, base', [
        (256, 32, 1024, 6666),
        (128, 64, 2345, 8888),
    ]
)
def test_qwen2_rope_embedding(seq_len, embed_dim, max_seq_len, base):
    x = torch.randn(1, seq_len, embed_dim)
    position_ids = torch.arange(seq_len)

    rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=embed_dim, base=base)
    nnm_cos = rope.cos[position_ids, :]
    nnm_sin = rope.sin[position_ids, :]
    # recover nnm sin embeds trick to normal format
    nnm_sin[:, :(embed_dim//2)] = -nnm_sin[:, :(embed_dim//2)]

    config = cfg.Qwen2Config(
        hidden_size=embed_dim, num_attention_heads=1, num_key_value_heads=1,
        rope_theta=base, max_position_embeddings=max_seq_len,
    )
    hf_rope = qwen2.Qwen2RotaryEmbedding(config=config)
    position_ids = position_ids.unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)

    assert nnm_cos.shape == hf_cos.shape and nnm_sin.shape == hf_sin.shape
    assert (nnm_cos - hf_cos).abs().mean() < 1e-5
    assert (nnm_sin - hf_sin).abs().mean() < 1e-5

@pytest.mark.parametrize(
    'seq_len, embed_dim, max_seq_len, base', [
        (256, 48, 1234, 7890),
        (128, 64, 2333, 9876),
    ]
)
def test_qwen2_apply_rope(seq_len, embed_dim, max_seq_len, base):
    x = torch.randn(1, seq_len, embed_dim)
    position_ids = torch.arange(seq_len)

    rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=embed_dim, base=base)
    nnm_y = rope(x)

    config = cfg.Qwen2Config(
        hidden_size=embed_dim, num_attention_heads=1, num_key_value_heads=1,
        rope_theta=base, max_position_embeddings=max_seq_len,
    )
    hf_rope = qwen2.Qwen2RotaryEmbedding(config=config)
    position_ids = position_ids.unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)
    hf_y, _ = qwen2.apply_rotary_pos_emb(x, x, hf_cos, hf_sin)
    hf_y = hf_y.squeeze(0)

    assert nnm_y.shape == hf_y.shape
    assert (nnm_y - hf_y).abs().mean() < 1e-5