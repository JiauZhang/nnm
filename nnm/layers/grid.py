from torch import nn


def patchify(input, patch_size):
    n, c, h, w = input.shape
    patches = input.reshape(n, c, h // patch_size, patch_size, w // patch_size, patch_size)
    return patches.permute(0, 2, 4, 3, 5, 1).reshape(n, -1, patch_size * patch_size * c)


def unpatchify(input, output_size, patch_size):
    n, _, _ = input.shape
    h, w = output_size
    c = input.shape[-1] // (patch_size * patch_size)
    patches = input.reshape(n, h // patch_size, w // patch_size, patch_size, patch_size, c)
    return patches.permute(0, 5, 1, 3, 2, 4).reshape(n, c, h, w)


class PatchEmbed2D(nn.Module):
    def __init__(self, *, kernel_size, in_channels, embed_dim, stride=None, padding=0, bias=False, norm=True, norm_eps=1e-6):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps) if norm else None

    def forward(self, input):
        output = self.proj(input)
        N, C, H, W = output.shape
        output = output.reshape(N, C, -1).permute(0, 2, 1)
        if self.norm is not None:
            output = self.norm(output)
        return output, (H, W)


class PatchEmbed3D(nn.Module):
    def __init__(self, *, in_channels, embed_dim, kernel_size, stride=None, padding=0, bias=False):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        N, C, D, H, W = x.shape
        x = x.reshape(N, C, -1).permute(0, 2, 1)
        return x, (D, H, W)
