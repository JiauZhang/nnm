import torch
from nnm.layers.attention import ChannelAttention, WindowAttention

def test_channel_attention():
    B, L, C = 4, 8, 32
    x = torch.randn(B, L, C)
    c_attn = ChannelAttention(C, 8)
    o = c_attn(x)
    assert list(o.shape) == [B, L, C]

def test_window_attention():
    B, H, W, C = 3, 16, 16, 32
    L = H * W
    x = torch.randn(B, L, C)
    w_attn = WindowAttention(C, 4, 8)
    o = w_attn(x)
    assert list(o.shape) == [B, L, C]
