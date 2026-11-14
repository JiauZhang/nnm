import torch

from tests.models.smoke_registry import SmokeCase, make, shape

# One case per architecture: scaling-only variants share a case; different block or
# composition architectures get separate arch entries.


def check_var_vqvae(model, output):
    reconstructed, codes = output
    assert tuple(reconstructed.shape) == (2, 3, 32, 32)
    assert tuple(codes.shape) == (2, 64)


def check_infinity(model, output):
    assert tuple(output.shape) == (2, 8, 4, 1)
    features = model.decode(torch.randint(0, 2, (2, 4)))
    assert tuple(features.shape) == (2, 16)


def check_jit(model, output):
    assert tuple(output.shape) == (2, 3, 32, 32)
    assert output.abs().max() <= 1.0
    unconditional = model(torch.randn(1, 3, 32, 32), torch.tensor([0.5]))
    assert tuple(unconditional.shape) == (1, 3, 32, 32)


CASES = [
    SmokeCase(
        "mmdit",
        "joint",
        make(
            "nnm.models.mmdit",
            "MMDiT",
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
        ),
        lambda: (
            torch.randn(2, 4, 32, 32),
            torch.randint(0, 1000, (2,)),
            torch.randn(2, 8, 32),
        ),
        shape(2, 4, 32, 32),
    ),
    SmokeCase(
        "var",
        "next-scale",
        make(
            "nnm.models.var",
            "VAR",
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=64,
            vocab_size=64,
            num_classes=10,
            num_scales=3,
            max_position_embeddings=16,
        ),
        lambda: (torch.randint(0, 64, (2, 8)), torch.tensor([1, 2]), torch.tensor([0, 1])),
        shape(2, 8, 64),
    ),
    SmokeCase(
        "var",
        "vqvae",
        make(
            "nnm.models.var",
            "VarVQVAE",
            hidden_size=16,
            in_channels=3,
            num_res_blocks=1,
            downsample=[2, 2],
            codebook_size=64,
            codebook_dim=8,
            norm_eps=1e-6,
        ),
        lambda: (torch.randn(2, 3, 32, 32),),
        check_var_vqvae,
    ),
    SmokeCase(
        "infinity",
        "bitwise",
        make(
            "nnm.models.infinity",
            "Infinity",
            hidden_size=32,
            feature_dim=16,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=64,
            num_classes=10,
            num_scales=3,
            num_quantizers=4,
            num_bits=1,
        ),
        lambda: (torch.randn(2, 8, 16), torch.tensor([1, 2]), torch.tensor([0, 1])),
        check_infinity,
    ),
    SmokeCase(
        "jit",
        "prefix",
        make(
            "nnm.models.jit",
            "JiT",
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
        ),
        lambda: (torch.randn(2, 3, 32, 32), torch.tensor([0.2, 0.7]), torch.tensor([1, 3])),
        check_jit,
    ),
]

CASES.append(
    SmokeCase(
        "bd3lm",
        "block-diffusion",
        make(
            "nnm.models.bd3lm",
            "Bd3lm",
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            cond_dim=16,
            vocab_size=100,
            model_length=16,
            block_size=4,
            cross_attn=True,
            adaln=True,
            causal=False,
            dropout=0.0,
            time_conditioning=False,
            var_min=True,
            sampling_eps_min=0.001,
            sampling_eps_max=0.999,
        ),
        lambda: (torch.randint(0, 100, (1, 32)), torch.rand(1)),
        shape(1, 16, 100),
    )
)
