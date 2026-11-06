import torch
from conippets.config import Config

from nnm.models.jit import JiT


def tiny_config():
    return Config(
        patch_size=4,
        data_dim=3,
        bottleneck_dim=16,
        dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        qk_norm=True,
        num_temb_freqs=16,
        num_classes=10,
    )


def test_jit_forward():
    torch.manual_seed(0)
    model = JiT(tiny_config()).eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0.2, 0.7])
    class_labels = torch.tensor([1, 3])
    with torch.no_grad():
        output = model(pixel_values, timesteps, class_labels)
    assert output.shape == (2, 3, 32, 32)
    assert output.abs().max() <= 1.0


def test_jit_unconditional():
    model = JiT(tiny_config()).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32), torch.tensor([0.5]))
    assert output.shape == (1, 3, 32, 32)
