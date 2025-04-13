import torch, pytest
from torch import nn
from nnm.layers.positional import RoPE, QwenRoPE

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

# https://github.com/huggingface/transformers/blob/953196a43dae6a3c474165fba7d215fcbc7b7730/src/transformers/models/qwen2/modeling_qwen2.py#L336
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, base, dim, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def qwen2_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def qwen2_apply_rotary_pos_emb(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (qwen2_rotate_half(q) * sin)
    return q_embed

@pytest.mark.parametrize(
    'seq_len, embed_dim, max_seq_len, base', [
        (256, 32, 1024, 6666),
        (128, 64, 1024, 8888),
        (512, 64, 1024, 8888),
        (888, 64, 1024, 8888),
    ]
)
def test_qwen2_rope(seq_len, embed_dim, max_seq_len, base):
    rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=embed_dim, base=base)
    x = torch.randn(1, seq_len, embed_dim)
    y = rope(x)
    rotary_emb = Qwen2RotaryEmbedding(base, embed_dim)
    position_ids = torch.arange(x.shape[-2]).reshape(1, -1)
    position_embeddings = rotary_emb(x, position_ids)
    cos, sin = position_embeddings
    qwen2_y = qwen2_apply_rotary_pos_emb(x, cos, sin).squeeze(dim=1)

    assert y.shape == qwen2_y.shape
    assert torch.abs(qwen2_y - y).mean() < 1e-5