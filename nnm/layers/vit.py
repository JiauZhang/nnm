import torch
from torch import nn
from torch.nn import functional as F_

def patchify(input, patch_size, dilation=1, padding=0, stride=None):
    stride = patch_size if stride is None else stride
    patches = F_.unfold(input, patch_size, dilation=dilation, padding=padding, stride=stride)
    # [N, C * patch_size^2, L] --> [N, L, C * patch_size^2]
    patches = patches.permute(0, 2, 1)
    return patches

def unpatchify(input, output_size, patch_size, dilation=1, padding=0, stride=None):
    stride = patch_size if stride is None else stride
    # [N, L, C * patch_size^2] --> [N, C * patch_size^2, L]
    input = input.permute(0, 2, 1)
    # [N, C * patch_size^2, L] --> [N, output_size[0], output_size[1], C]
    output = F_.fold(input, output_size, patch_size, dilation=dilation, padding=padding, stride=stride)
    return output

class PatchEmbed2D(nn.Module):
    def __init__(self, *, kernel_size, in_channels, embed_dim, stride=None, padding=0, bias=False):
        super().__init__()
        if stride is None: stride = kernel_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input):
        output = self.proj(input)
        N, C, H, W = output.shape
        output = output.reshape(N, C, -1).permute(0, 2, 1)
        output = self.norm(output)
        return output, (H, W)

class PatchEmbed3D(nn.Module):
    def __init__(self, *, in_channels, embed_dim, kernel_size, stride=None, padding=0, bias=False):
        super().__init__()
        if stride is None: stride = kernel_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        N, C, D, H, W = x.shape
        x = x.reshape(N, C, -1).permute(0, 2, 1)
        return x, (D, H, W)
