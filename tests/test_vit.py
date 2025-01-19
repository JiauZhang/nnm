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
