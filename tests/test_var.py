import torch
from conippets.config import Config

from nnm.models.var import VAR, VarVQVAE


def tiny_config():
    return Config(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=64,
        num_classes=10,
        num_scales=3,
        max_position_embeddings=16,
    )


def test_var_forward():
    torch.manual_seed(0)
    model = VAR(tiny_config()).eval()
    token_ids = torch.randint(0, 64, (2, 8))
    class_labels = torch.tensor([1, 2])
    scale = torch.tensor([0, 1])
    with torch.no_grad():
        logits = model(token_ids, class_labels, scale)
    assert logits.shape == (2, 8, 64)


def test_var_vqvae():
    torch.manual_seed(0)
    config = Config(
        hidden_size=16,
        in_channels=3,
        num_res_blocks=1,
        downsample=[2, 2],
        codebook_size=64,
        codebook_dim=8,
        norm_eps=1e-6,
    )
    model = VarVQVAE(config).eval()
    with torch.no_grad():
        reconstructed, codes = model(torch.randn(2, 3, 32, 32))
    assert reconstructed.shape == (2, 3, 32, 32)
    assert codes.shape == (2, 64)
