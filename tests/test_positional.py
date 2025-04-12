import torch, pytest
from nnm.layers.positional import RoPE

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
    'seq_len, embed_dim, max_seq_len, base', [
        (256, 32, 1024, 6666),
        (128, 64, 1024, 10000),
        (512, 64, 1024, 10000),
        (999, 64, 1024, 10000),
    ]
)
def test_rope(seq_len, embed_dim, max_seq_len, base):
    rope = RoPE(max_seq_len=max_seq_len, embed_dim=embed_dim, base=base)
    x = torch.randn(1, seq_len, embed_dim)
    y = rope(x)
    llama_freqs_cis = llama_precompute_freqs_cis(embed_dim, max_seq_len, base)[:seq_len]
    llama_y = llama_apply_rotary_emb(x, llama_freqs_cis)
    llama_y = llama_y.flatten(-2)

    assert y.shape == llama_y.shape
    assert torch.abs(llama_y - y).mean() < 1e-5
