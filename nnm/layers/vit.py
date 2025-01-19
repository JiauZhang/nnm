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

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_channels, embed_dim, padding=0):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input):
        output = self.proj(input)
        N, C, H, W = output.shape[:2]
        output = output.reshape(N, C, -1).permute(0, 2, 1)
        output = self.norm(output)
        return output, (H, W)
