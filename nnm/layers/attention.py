from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, dim, groups, qkv_bias=True):
        super().__init__()
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        B, L, C = x.shape
        assert C % self.groups == 0, "input feature has wrong channel"
        # [B, L, C] --> [B, L, 3C] --> [B, L, 3, g, C//g] --> [3, B, g, L, C//g]
        qkv = self.qkv(x).reshape(B, L, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        # [B, g, L, C//g]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * (float(L) ** -0.5)
        # [B, g, C//g, C//g]
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        # [B, g, C//g, C//g] @ [B, g, C//g, L] --> [B, g, C//g, L] --> [B, g, L, C//g]
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        # [B, g, L, C//g] --> [B, L, g, C//g] --> [B, L, C]
        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x, size

def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, batch_size: int, window_size: int, H: int, W: int):
    B = batch_size
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.reshape(-1, self.window_size * self.window_size, C)

        B_, L, C = x.shape
        qkv = self.qkv(x).reshape(B_, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, L, C)
        x = self.proj(x)

        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, B, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, H * W, C)

        return x, size
