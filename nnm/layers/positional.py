import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, *, max_seq_len, embed_dim, base=10000):
        super().__init__()
        assert (embed_dim % 2) == 0, 'embed_dim must be divided by 2'
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.base = base
        self.precompute()

    @torch.no_grad()
    def precompute(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.embed_dim, 2, dtype=torch.float64) / self.embed_dim))
        theta = theta.reshape(-1, 1).repeat(1, 2).reshape(-1)
        m = torch.arange(self.max_seq_len, device=theta.device, dtype=torch.float64)
        m_theta = torch.outer(m, theta)
        self.cos = torch.cos(m_theta).to(dtype=torch.float32)
        # [-1, 1, -1, 1, ...]
        m_theta = (m_theta.reshape(-1, 2) * torch.tensor([-1, 1], dtype=torch.float64)).reshape(
            self.max_seq_len, self.embed_dim
        )
        self.sin = torch.sin(m_theta).to(dtype=torch.float32)

    def seq_idx(self, cache, position, seq_len):
        if cache:
            assert position is not None
            seq_idx = torch.arange(position, position + seq_len)
        else:
            seq_idx = torch.arange(seq_len)
        return seq_idx

    def forward(self, x, cache=False, position=None):
        # [..., seq_len, embed_dim]
        shape = x.shape
        assert shape[-1] == self.embed_dim
        seq_idx = self.seq_idx(cache, position, shape[-2])
        sin_pe = self.sin[seq_idx, :]
        cos_pe = self.cos[seq_idx, :]
        y = x * cos_pe + x.reshape(-1, 2).flip(dims=[-1]).reshape(shape) * sin_pe
        return y

class QwenRoPE(RoPE):
    @torch.no_grad()
    def precompute(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.embed_dim, 2, dtype=torch.float64) / self.embed_dim))
        m = torch.arange(self.max_seq_len, device=theta.device, dtype=torch.float64)
        # [max_seq_len, embed_dim // 2]
        m_theta = torch.outer(m, theta)
        # [max_seq_len, embed_dim]
        self.sin = torch.sin(torch.cat([-m_theta, m_theta], dim=-1)).to(dtype=torch.float32)
        self.cos = torch.cos(m_theta).to(dtype=torch.float32).repeat(1, 2)

    def forward(self, x, cache=False, position=None):
        shape = x.shape
        assert shape[-1] == self.embed_dim
        seq_idx = self.seq_idx(cache, position, shape[-2])
        sin_pe = self.sin[seq_idx, :]
        cos_pe = self.cos[seq_idx, :]
        half_embed_dim = shape[-1] // 2
        x1 = x[..., :half_embed_dim]
        x2 = x[..., half_embed_dim:]
        y = torch.cat((x2, x1), dim=-1)
        sin_pe = self.sin[seq_idx, :]
        cos_pe = self.cos[seq_idx, :]
        y = (x * cos_pe) + (y * sin_pe)
        return y
