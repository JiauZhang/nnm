import torch
from conippets.config import Config
from transformers import Florence2VisionBackbone, Florence2VisionConfig

from nnm.models.davit import DaViT


def test_davit_matches_transformers():
    config = Config(
        depths=[1, 1],
        patch_size=[7, 3],
        patch_stride=[4, 2],
        patch_padding=[3, 1],
        patch_prenorm=[False, True],
        embed_dim=[8, 16],
        embed_dims=[8, 16],
        num_heads=[1, 2],
        num_groups=[1, 2],
        window_size=7,
        drop_path_rate=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    torch.manual_seed(0)
    hf_model = Florence2VisionBackbone(Florence2VisionConfig(**config)).eval()
    nnm_model = DaViT(config).eval()
    nnm_model.load_state_dict(hf_model.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = hf_model(pixels).last_hidden_state
        actual = nnm_model(pixels)
    assert torch.allclose(actual, expected, atol=1e-5)
