import torch
from nnm.layers.attention import ChannelAttention, WindowAttention

def test_channel_attention():
    B, H, W, C = 4, 4, 3, 32
    B, L, C = 4, H * W, 32
    groups = 8
    x = torch.randn(B, L, C)
    c_attn = ChannelAttention(C, groups)
    o, size = c_attn(x, (H, W))
    assert list(o.shape) == [B, L, C]

def test_window_attention():
    B, H, W, C = 3, 16, 16, 32
    L = H * W
    x = torch.randn(B, L, C)
    w_attn = WindowAttention(C, 4, 8)
    o, size = w_attn(x, (H, W))
    assert list(o.shape) == [B, L, C]
