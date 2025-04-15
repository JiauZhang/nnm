import torch
from torch import nn

class RWKVChannelMix(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        hidden_sz = 4 * embed_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_k = None
        self.time_mix_r = None
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
        self.time_decay = None
        self.time_first = None
        self.time_mix_k = None
        self.time_mix_v = None
        self.time_mix_r = None
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

    def wkv(self, w, u, k, v):
        # w and u shape: [1, 1, embed_dim]
        w = -w.exp()

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