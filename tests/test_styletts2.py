import pytest
import torch
from conippets.config import Config

from nnm.models.styletts2 import StyleTTS2


def _config(multispeaker=True):
    return Config(
        hidden_dim=64,
        style_dim=16,
        n_layer=2,
        n_token=178,
        n_mels=80,
        max_dur=50,
        dropout=0.2,
        dim_in=16,
        max_conv_dim=64,
        multispeaker=multispeaker,
        sample_rate=24000,
        albert=dict(
            vocab_size=178,
            embedding_size=32,
            hidden_size=64,
            num_hidden_layers=2,
            num_hidden_groups=1,
            inner_group_num=1,
            num_attention_heads=4,
            intermediate_size=128,
            layer_norm_eps=1e-12,
            max_position_embeddings=64,
            type_vocab_size=2,
            hidden_act="gelu_new",
        ),
        decoder=dict(
            type="hifigan",
            resblock_kernel_sizes=[3, 7, 11],
            upsample_rates=[10, 5, 3, 2],
            upsample_initial_channel=64,
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_kernel_sizes=[20, 10, 6, 4],
        ),
        diffusion=dict(
            embedding_mask_proba=0.1,
            transformer=dict(num_layers=2, num_heads=4, head_features=16, multiplier=2),
            dist=dict(mean=-3.0, std=1.0, sigma_data=0.2),
        ),
    )


@pytest.mark.parametrize("multispeaker", [True, False])
def test_infer(multispeaker):
    model = StyleTTS2(_config(multispeaker)).eval()
    ids = torch.randint(0, 178, (1, 12))
    style = torch.randn(1, 32)
    out = model.infer(ids, style, speed=1.0)
    assert out["waveform"].dim() == 1
    assert out["waveform"].numel() > 0
    assert out["duration"].shape == (12,)
    assert (out["duration"] >= 1).all()


def test_extract_style():
    model = StyleTTS2(_config()).eval()
    mel = torch.randn(1, 1, 80, 128)
    acoustic, prosodic = model.extract_style(mel)
    assert acoustic.shape == (1, 16)
    assert prosodic.shape == (1, 16)


@pytest.mark.parametrize("multispeaker", [True, False])
def test_sample_style(multispeaker):
    model = StyleTTS2(_config(multispeaker)).eval()
    bert_embedding = torch.randn(1, 10, 64)
    reference = torch.randn(1, 32) if multispeaker else None
    style = model.sample_style(bert_embedding, reference=reference, num_steps=3)
    assert style.shape == (1, 32)
    assert torch.isfinite(style).all()


def test_diffusion_loss():
    model = StyleTTS2(_config()).train()
    x = torch.randn(2, 1, 32)
    bert_embedding = torch.randn(2, 10, 64)
    reference = torch.randn(2, 32)
    loss = model.diffusion(x, embedding=bert_embedding, features=reference)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
