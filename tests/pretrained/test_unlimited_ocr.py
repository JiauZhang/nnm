import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conippets.config import Config

from nnm.layers.rope import RoPE2D


def tiny_config():
    return Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="softmax",
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=500000.0,
        pad_token_id=0,
        rope_scaling={
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "attention_factor": 1.0,
            "rope_type": "yarn",
            "type": "yarn",
        },
        projector_config={"input_dim": 32, "n_embed": 16, "projector_type": "linear"},
        vision_config={
            "image_size": 56,
            "width": {
                "sam_vit_b": {
                    "image_size": 56,
                    "patch_size": 7,
                    "width": 16,
                    "layers": 2,
                    "heads": 2,
                    "mlp_ratio": 2.0,
                    "out_chans": 8,
                    "window_size": 4,
                    "global_attn_indexes": [1],
                    "downsample_channels": [16, 16],
                    "norm_eps": 1e-6,
                },
                "clip-l-14-224": {
                    "image_size": 14,
                    "patch_size": 7,
                    "width": 16,
                    "layers": 2,
                    "heads": 2,
                    "ffn_hidden_size": 32,
                    "pre_layernorm_epsilon": 1e-5,
                    "layernorm_epsilon": 1e-5,
                    "mlp_ratio": 2.0,
                },
            },
        },
        sparse_attention_config={
            "sparse_block_size": 2,
            "sparse_local_block": 1,
            "sparse_index_dim": 8,
            "sparse_num_index_heads": 1,
            "sparse_topk_blocks": 3,
            "dash_topk_blocks": 3,
            "dash_scaling_factor": 1.0,
            "dash_floor_per_layer": [1e-4, 1e-4],
            "dash_scale_per_layer": [1.0, 1.0],
            "dash_sigma": 1.0,
            "dash_prior_floor": 1e-4,
            "dash_stage0": "head_cls",
            "dash_query_chunk": 8,
            "dash_image_only": True,
            "image_bucket_constraint": True,
            "image_token_id": 61,
            "page_token_id": 60,
            "per_page_image_tokens": 2,
            "sparse_main_rope_mode": "page2d",
            "sparse_rope_tok_theta": 10000.0,
            "sparse_rope_page_theta": 256.0,
            "sparse_rope_page_pairs": 1,
            "sparse_rope_page_mod": 2,
            "sparse_main_rope_tok_freq_mode": "resample",
            "sparse_rope_theta": 500000.0,
        },
    )


def tiny_page_ids():
    return torch.tensor([[0, 60] + [61] * 7 + [60] + [61] * 7 + [60, 62, 63, 10]])


def expected_parameter_count(config):
    vision = config.vision_config
    sam = vision["width"]["sam_vit_b"]
    clip = vision["width"]["clip-l-14-224"]
    width = sam.width
    grid = sam.image_size // sam.patch_size
    head_dim = width // sam.heads
    mlp_dim = int(width * sam.mlp_ratio)
    per_window_block = (
        4 * width
        + width * 3 * width
        + 3 * width
        + width * width
        + width
        + width * mlp_dim
        + mlp_dim
        + mlp_dim * width
        + width
        + (2 * sam.window_size - 1) * head_dim * 2
    )
    per_global_block = per_window_block - (2 * sam.window_size - 1) * head_dim * 2 + (2 * grid - 1) * head_dim * 2
    window_blocks = sam.layers - len(sam.global_attn_indexes)
    sam_params = (
        3 * width * sam.patch_size**2
        + width
        + grid**2 * width
        + window_blocks * per_window_block
        + len(sam.global_attn_indexes) * per_global_block
        + width * sam.out_chans
        + 2 * sam.out_chans
        + sam.out_chans * sam.out_chans * 9
        + 2 * sam.out_chans
        + sam.out_chans * sam.downsample_channels[0] * 9
        + sam.downsample_channels[0] * sam.downsample_channels[1] * 9
    )
    cw = clip.width
    ffn = clip.ffn_hidden_size
    positions = (clip.image_size // clip.patch_size) ** 2 + 1
    per_clip_layer = 4 * cw + (3 * cw * cw + 3 * cw) + (cw * cw + cw) + (cw * ffn + ffn) + (ffn * cw + cw)
    clip_params = cw + 3 * cw * clip.patch_size**2 + positions * cw + 2 * cw + clip.layers * per_clip_layer
    projector = config.projector_config
    projector_params = projector.input_dim * projector.n_embed + projector.n_embed
    hidden = config.hidden_size
    index_dim = config.sparse_attention_config.sparse_index_dim
    attention = 4 * hidden * hidden
    router = 2 * hidden * index_dim + 2 * index_dim + index_dim
    norms = 2 * hidden
    dense_mlp = 2 * hidden * config.intermediate_size + config.intermediate_size * hidden
    moe = (
        config.n_routed_experts * hidden
        + config.n_routed_experts * 3 * hidden * config.moe_intermediate_size
        + 3 * hidden * config.moe_intermediate_size * config.n_shared_experts
    )
    dense_layers = config.first_k_dense_replace
    moe_layers = config.num_hidden_layers - dense_layers
    decoder_params = (
        config.vocab_size * hidden
        + config.vocab_size * hidden
        + 3 * hidden
        + config.num_hidden_layers * (norms + attention + router)
        + dense_layers * dense_mlp
        + moe_layers * moe
    )
    return sam_params + clip_params + projector_params + decoder_params


def page_rope(dim=8, tok_theta=10000.0, page_theta=256.0, page_pairs=1, tok_freq_mode="resample"):
    half = dim // 2
    n_tok = half - page_pairs
    positions = torch.arange(0, n_tok, dtype=torch.float32)
    if tok_freq_mode == "truncate":
        inv_tok = 1.0 / (tok_theta ** (positions * 2 / dim))
    else:
        inv_tok = 1.0 / (tok_theta ** (positions / n_tok))
    inv_page = 1.0 / (page_theta ** (torch.arange(0, page_pairs, dtype=torch.float32) / page_pairs))
    return RoPE2D([inv_tok, inv_page])


def page_positions(tok, page, mod=0):
    if mod > 0:
        page = torch.remainder(page, mod)
    return torch.stack([tok, page], dim=-1)


@pytest.mark.pretrained
def test_upstream_deepencoder_reference(require_reference_path):
    if not os.path.isdir(require_reference_path):
        pytest.skip("--reference-path with the upstream source directory is required")
    upstream_python = os.environ.get("UNLIMITED_OCR_PYTHON", sys.executable)
    script = f"""
import importlib.util
import sys
import types
from functools import partial

import torch

einops = types.ModuleType("einops")
einops.rearrange = lambda *args, **kwargs: None
einops.repeat = lambda *args, **kwargs: None
sys.modules["einops"] = einops

class EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

easydict = types.ModuleType("easydict")
easydict.EasyDict = EasyDict
sys.modules["easydict"] = easydict

try:
    spec = importlib.util.spec_from_file_location("upstream_deepencoder", {os.path.join(require_reference_path, "deepencoder.py")!r})
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
except Exception as exc:
    print("UPSTREAM_IMPORT_ERROR", repr(exc))
    raise SystemExit(2)

from conippets.config import Config
from nnm.models.clip import CLIPVisionModel
from nnm.models.sam import SamVisionEncoder


pytestmark = pytest.mark.pretrained

torch.manual_seed(0)
up_sam = upstream.ImageEncoderViT(
    depth=2, embed_dim=16, img_size=56, mlp_ratio=1,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=2, patch_size=7,
    qkv_bias=True, use_rel_pos=True, global_attn_indexes=[1], window_size=4, out_chans=256,
)


class OurSam(SamVisionEncoder):
    def __init__(self):
        super().__init__(
            Config(
                hidden_size=16, num_attention_heads=2, image_size=56, patch_size=7,
                num_hidden_layers=2, mlp_ratio=1, output_channels=256,
                window_size=4, global_attn_indexes=[1], layer_norm_eps=1e-6,
            )
        )
        self.downsample_2x = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample_4x = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, pixel_values):
        return self.downsample_4x(self.downsample_2x(super().forward(pixel_values)))


ours = OurSam()
mapped = dict()
for key, tensor in up_sam.state_dict().items():
    key = key.replace("patch_embed.projection.", "patch_embed.proj.")
    key = key.replace("blocks.", "layers.")
    key = key.replace(".mlp.lin1.", ".mlp.fc1.").replace(".mlp.lin2.", ".mlp.fc2.")
    key = key.replace("net_2.", "downsample_2x.").replace("net_3.", "downsample_4x.")
    mapped[key] = tensor
ours.load_state_dict(mapped, strict=True)
pixel_values = torch.randn(2, 3, 56, 56)
torch.testing.assert_close(up_sam(pixel_values), ours(pixel_values), atol=1e-4, rtol=1e-4)

clip_cfg = upstream.adict(
    num_layers=2, hidden_size=16, num_heads=2, num_attention_heads=2, ffn_hidden_size=32,
    seq_length=4, max_position_embeddings=4, use_flash_attn=False,
    understand_projector_stride=2, hidden_dropout=0.0, attention_dropout=0.0,
    no_persist_layer_norm=False, layernorm_epsilon=1e-5, pre_layernorm_epsilon=1e-5,
    image_size=14, patch_size=7, recompute_list=[],
)
up_clip = upstream.VitModel(cfg=clip_cfg, freeze_embed=False, freeze_pre_norm=False)
ours_clip = CLIPVisionModel(
    Config(
        hidden_size=16, num_attention_heads=2, num_hidden_layers=2, intermediate_size=32,
        image_size=14, patch_size=7, layer_norm_eps=1e-5, hidden_act="quick_gelu",
    ),
    post_layernorm=False,
)
mapped = dict()
for key, tensor in up_clip.state_dict().items():
    if key.endswith("self_attn.qkv_proj.weight") or key.endswith("self_attn.qkv_proj.bias"):
        prefix = key[: -len("qkv_proj.weight" if key.endswith("weight") else "qkv_proj.bias")]
        suffix = "weight" if key.endswith("weight") else "bias"
        for name, part in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
            mapped[prefix + name + "." + suffix] = part.contiguous()
        continue
    key = key.replace("transformer.layers.", "encoder.layers.")
    key = key.replace(".layer_norm1.", ".norm1.").replace(".layer_norm2.", ".norm2.")
    key = key.replace(".self_attn.out_proj.", ".attn.o_proj.")
    key = key.replace(".self_attn.", ".attn.")
    mapped[key] = tensor
ours_clip.load_state_dict(mapped, strict=True)
patch_embeds = torch.randn(2, 16, 2, 2)
torch.testing.assert_close(
    up_clip(pixel_values, patch_embeds),
    ours_clip(pixel_values, patch_embeds=patch_embeds)["last_hidden_state"],
    atol=1e-4,
    rtol=1e-4,
)
patch_embeds = torch.randn(2, 16, 3, 3)
torch.testing.assert_close(
    up_clip(pixel_values, patch_embeds),
    ours_clip(pixel_values, patch_embeds=patch_embeds)["last_hidden_state"],
    atol=1e-4,
    rtol=1e-4,
)
print("UPSTREAM_OK")
"""
    result = subprocess.run(
        [upstream_python, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    if result.returncode == 2:
        pytest.skip(f"upstream dynamic code cannot run: {result.stdout.strip()} {result.stderr.strip()}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UPSTREAM_OK" in result.stdout
