import torch
from conippets.config import Config
from transformers import Florence2VisionBackbone, Florence2VisionConfig

from nnm.models.davit import DaViT
from nnm.models.florence import Florence2


def test_davit_matches_transformers():
    config = Config(
        depths=[1, 1],
        patch_size=[7, 3],
        patch_stride=[4, 2],
        patch_padding=[3, 1],
        patch_prenorm=[False, True],
        embed_dim=[8, 16],
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


def tiny_config():
    return Config(
        image_token_id=5,
        vision_config=dict(
            image_size=32,
            in_channels=3,
            embed_dim=[8, 16],
            depths=[1, 1],
            num_heads=[1, 2],
            num_groups=[1, 2],
            patch_size=[7, 3],
            patch_stride=[4, 2],
            patch_padding=[3, 1],
            patch_prenorm=[False, True],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            projection_dim=16,
            max_position_embeddings=32,
            max_temporal_embeddings=8,
        ),
        text_config=dict(
            hidden_size=16,
            num_attention_heads=4,
            intermediate_size=32,
            vocab_size=64,
            max_position_embeddings=32,
            encoder_layers=1,
            decoder_layers=1,
            layer_norm_eps=1e-5,
        ),
    )


def test_florence2_forward():
    torch.manual_seed(0)
    model = Florence2(tiny_config()).eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 64, (2, 17))
    input_ids[:, :17] = 5
    decoder_input_ids = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        logits = model(pixel_values, input_ids, decoder_input_ids=decoder_input_ids)
    assert logits.shape == (2, 6, 64)


def test_florence2_image_features():
    model = Florence2(tiny_config()).eval()
    with torch.no_grad():
        features = model.get_image_features(torch.randn(1, 3, 32, 32))
    assert features.shape == (1, 17, 16)
