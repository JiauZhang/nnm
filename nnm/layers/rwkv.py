import torch
from torch import nn

class RWKVChannelMix(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        hidden_sz = 4 * embed_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(embed_dim, hidden_sz, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(hidden_sz, embed_dim, bias=False)
        self.init_weight()

    @torch.no_grad()
    def init_weight(self):
        scales = torch.arange(self.embed_dim, dtype=torch.float32).reshape(1, 1, -1) / self.embed_dim
        self.time_mix_k = nn.Parameter(scales.clone())
        self.time_mix_r = nn.Parameter(scales.clone())
        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x_t):
        x_t_1 = self.time_shift(x_t)

        xk = torch.lerp(x_t_1, x_t, self.time_mix_k)
        k = self.key(xk).relu().square()
        kv = self.value(k)

        xr = torch.lerp(x_t_1, x_t, self.time_mix_r)
        rkv = self.receptance(xr).sigmoid() * kv
        return rkv

class RWKVTimeMix(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.init_weight()

    @torch.no_grad()
    def init_weight(self):
        scales = torch.arange(self.embed_dim, dtype=torch.float32).reshape(1, 1, -1) / self.embed_dim
        self.time_decay = nn.Parameter(scales.clone())
        self.time_first = nn.Parameter(scales.clone())
        self.time_mix_k = nn.Parameter(scales.clone())
        self.time_mix_v = nn.Parameter(scales.clone())
        self.time_mix_r = nn.Parameter(scales.clone())
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    # eq. (19)-(22)
    def wkv_raw(self, w, u, k, v):
        # w and u shape: [1, 1, embed_dim], k and v shape: [batch, seq_len, embed_dim]
        w_exp = -w.exp()
        wkv = torch.empty_like(k)
        a = 0
        b = 0
        for idx in range(k.shape[-2]):
            k_t = k[:, idx]
            v_t = v[:, idx]

            k_exp = k_t.exp()
            uk_exp = (u + k_t).exp()
            wkv[:, idx] = (a + uk_exp * v_t) / (b + uk_exp)
            a = w_exp * a + k_exp * v_t
            b = w_exp * b + k_exp
        return wkv

    # eq. (23)-(28) for numerical safe version
    def wkv(self, w, u, k, v):
        w_exp = -w.exp()
        wkv = torch.empty_like(k)
        p = 0
        q = 0
        o = torch.tensor(float('-inf'), dtype=u.dtype, device=u.device)
        for idx in range(k.shape[-2]):
            k_t = k[:, idx]
            v_t = v[:, idx]

            uk = u + k_t
            no = torch.max(o, uk)
            A = (o - no).exp()
            B = (uk - no).exp()
            wkv[:, idx] = (A * p + B * v_t) / (A * q + B)

            wo = w_exp + o
            no = torch.max(wo, k_t)
            A = (wo - no).exp()
            B = (k_t - no).exp()
            p = A * p + B * v_t
            q = A * q + B
            o = no
        return wkv

    def forward(self, x_t):
        x_t_1 = self.time_shift(x_t)
        xk = torch.lerp(x_t_1, x_t, self.time_mix_k)
        xv = torch.lerp(x_t_1, x_t, self.time_mix_v)
        xr = torch.lerp(x_t_1, x_t, self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        sr = self.receptance(xr).sigmoid()

        rwkv = sr * self.wkv(self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv