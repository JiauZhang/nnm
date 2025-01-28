import torch
from nnm.layers import vit

def test_patchify():
    feature = torch.randn(4, 16, 32, 64)
    patches = vit.patchify(feature, 8)
    assert list(patches.shape) == [4, 32//8 * 64//8, 16 * 8 * 8]
    # patch at (1, 2) of (4, 8)
    patch = feature[:, :, 8:16, 16:24].reshape(4, -1)
    assert bool((patches[:, 1 * 8 + 2] == patch).all())

def test_unpatchify():
    hidden_state = torch.randn(4, 32, 16 * 8 * 8)
    feature = vit.unpatchify(hidden_state, (32, 64), 8)
    assert list(feature.shape) == [4, 16, 32, 64]
    # patch at (2, 6) of (4, 8)
    patch = hidden_state[:, 22]
    feature_patch = feature[:, :, 16:24, 48:56].reshape(4, -1)
    assert bool((feature_patch == patch).all())

def test_patch_embed_2d():
    image = torch.randn(4, 3, 256, 128)
    patch_embed = vit.PatchEmbed2D(kernel_size=7, stride=4, in_channels=3, embed_dim=16, padding=3)
    embeds, size = patch_embed(image)
    h, w = 256 // 4, 128 // 4
    assert size == (h, w)
    assert list(embeds.shape) == [4, h * w, 16]

def test_patch_embed_3d():
    N, C, D, H, W = 4, 3, 8, 128, 128
    images = torch.randn(N, C, D, H, W)
    patch_embed = vit.PatchEmbed3D(kernel_size=(D, 16, 16), in_channels=C, embed_dim=16)
    embeds, size = patch_embed(images)
    d, h, w = 1, 128 // 16, 128 // 16
    assert size == (d, h, w)
    assert list(embeds.shape) == [4, d * h * w, 16]
