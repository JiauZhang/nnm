import math, torch
from torch import nn

def get_seq_idx(seq_len, kv_cache=False, position=None):
    if kv_cache:
        seq_idx = torch.arange(position, position + seq_len)
    else:
        seq_idx = torch.arange(seq_len)
    return seq_idx

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

    def forward(self, x, kv_cache=False, position=None):
        # [..., seq_len, embed_dim]
        shape = x.shape
        assert shape[-1] == self.embed_dim
        seq_idx = get_seq_idx(shape[-2], kv_cache, position)
        sin_pe = self.sin[seq_idx, :]
        cos_pe = self.cos[seq_idx, :]
        y = x * cos_pe + x.reshape(-1, 2).flip(dims=[-1]).reshape(shape) * sin_pe
        return y

class QwenRoPE(nn.Module):
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
        m = torch.arange(self.max_seq_len, device=theta.device, dtype=torch.float64)
        # [max_seq_len, embed_dim // 2]
        m_theta = torch.outer(m, theta)
        # [max_seq_len, embed_dim]
        sin_m_theta = torch.sin(m_theta)
        # pre-negative sin embeds
        # [-x2, x1] * [sin_m_theta, sin_m_theta] --> [x2, x1] * [-sin_m_theta, sin_m_theta]
        self.sin = torch.cat([-sin_m_theta, sin_m_theta], dim=-1).to(dtype=torch.float32)
        self.cos = torch.cos(m_theta).to(dtype=torch.float32).repeat(1, 2)

    def forward(self, x, kv_cache=False, position=None):
        shape = x.shape
        assert shape[-1] == self.embed_dim
        seq_idx = get_seq_idx(shape[-2], kv_cache, position)
        sin_pe = self.sin[seq_idx, :]
        cos_pe = self.cos[seq_idx, :]
        half_embed_dim = shape[-1] // 2
        x1 = x[..., :half_embed_dim]
        x2 = x[..., half_embed_dim:]
        y = torch.cat((x2, x1), dim=-1)
        y = (x * cos_pe) + (y * sin_pe)
        return y

# inverse embed_dim based on eq. (17)
def find_correction_dim(num_rotations, embed_dim, base=10000, max_seq_len=2048):
    return (embed_dim * math.log(max_seq_len/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def find_correction_range(low_rot, high_rot, embed_dim, base=10000, max_seq_len=2048):
    low = math.floor(find_correction_dim(low_rot, embed_dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, embed_dim, base, max_seq_len))
    return max(low, 0), min(high, embed_dim-1)  # Clamp values just in case

def linear_ramp_mask(min, max, embed_dim):
    if min == max: max += 0.001
    linear_func = (torch.arange(embed_dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def get_temperature(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class YaRN(nn.Module):
    def __init__(self, *, max_seq_len, embed_dim, base=10000, scale=1, alpha=1, beta=32):
        super().__init__()
        assert (embed_dim % 2) == 0, 'embed_dim must be divided by 2'
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.base = base
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
        self.precompute()

    @torch.no_grad()
    def precompute(self):
        pos_freqs = self.base ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = find_correction_range(self.beta, self.alpha, self.embed_dim, self.base, self.max_seq_len)
        # n-d rotational scaling corrected for extrapolation
        inv_freq_mask = 1 - linear_ramp_mask(low, high, self.embed_dim // 2).float()
        theta = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.temperature = float(get_temperature(self.scale))

        m = torch.arange(self.max_seq_len * self.scale, dtype=torch.float64)
        m_theta = torch.outer(m, theta)
        sin_m_theta = torch.sin(m_theta) * self.temperature
        self.sin = torch.cat([-sin_m_theta, sin_m_theta], dim=-1).to(dtype=torch.float32)
        self.cos = (torch.cos(m_theta) * self.temperature).to(dtype=torch.float32).repeat(1, 2)

    def forward(self, x, kv_cache=False, position=None):
        shape = x.shape
        assert shape[-1] == self.embed_dim
        seq_idx = get_seq_idx(shape[-2], kv_cache, position)
        sin_pe = self.sin[seq_idx, :]
        cos_pe = self.cos[seq_idx, :]
        half_embed_dim = shape[-1] // 2
        x1 = x[..., :half_embed_dim]
        x2 = x[..., half_embed_dim:]
        y = torch.cat((x2, x1), dim=-1)
        y = (x * cos_pe) + (y * sin_pe)
        return y