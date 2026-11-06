import torch
from conippets.config import Config

from nnm.models.mmdit import MMDiT


def tiny_config():
    return Config(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=4,
        in_channels=4,
        context_dim=32,
        mlp_ratio=4.0,
        qkv_bias=True,
        o_bias=True,
        qk_norm=True,
        norm_eps=1e-6,
        image_size=32,
    )


def test_mmdit_forward():
    torch.manual_seed(0)
    model = MMDiT(tiny_config()).eval()
    hidden_states = torch.randn(2, 4, 32, 32)
    timestep = torch.randint(0, 1000, (2,))
    context = torch.randn(2, 8, 32)
    with torch.no_grad():
        output = model(hidden_states, timestep, context)
    assert output.shape == (2, 4, 32, 32)
