import torch
from conippets.config import Config

from nnm.models.infinity import Infinity


def tiny_config():
    return Config(
        hidden_size=32,
        feature_dim=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        num_classes=10,
        num_scales=3,
        num_quantizers=4,
        num_bits=1,
    )


def test_infinity_forward():
    torch.manual_seed(0)
    model = Infinity(tiny_config()).eval()
    features = torch.randn(2, 8, 16)
    class_labels = torch.tensor([1, 2])
    scale = torch.tensor([0, 1])
    with torch.no_grad():
        logits = model(features, class_labels, scale)
    assert logits.shape == (2, 8, 4, 1)


def test_infinity_decode():
    model = Infinity(tiny_config()).eval()
    codes = torch.randint(0, 2, (2, 4))
    with torch.no_grad():
        features = model.decode(codes)
    assert features.shape == (2, 16)
