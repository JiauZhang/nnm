import os
import subprocess
import sys

import pytest
import torch

from conippets.config import Config
from nnm.layers.rope import RoPE2D, apply_rope
from nnm.models.unlimited_ocr import (
    BlockRouter,
    ClipEncoder,
    MoE,
    SamEncoder,
    UnlimitedOCR,
    UnlimitedOCRModel,
    build_candidate_mask,
    build_page_aware_ids,
    build_page_rope_pos,
    entmax15,
    sparse_keep_mask,
)


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
    projections = 2 * hidden * index_dim + 2 * index_dim
    router = projections + index_dim
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
        + config.num_hidden_layers * (norms + attention + projections + router)
        + dense_layers * dense_mlp
        + moe_layers * moe
    )
    return sam_params + clip_params + projector_params + decoder_params


def test_parameter_count_matches_config(weight_config):
    config = weight_config("Unlimited-OCR-2")
    with torch.device("meta"):
        model = UnlimitedOCR(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == expected_parameter_count(config)
    assert total == 3_343_978_240


def test_sam_encoder_shapes():
    encoder = SamEncoder(
        image_size=56,
        patch_size=7,
        width=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        out_chans=8,
        window_size=4,
        global_attn_indexes=[1],
        downsample_channels=(16, 16),
        norm_eps=1e-6,
    )
    pixel_values = torch.randn(2, 3, 56, 56)
    output = encoder(pixel_values)
    assert output.shape == (2, 16, 2, 2)
    assert torch.isfinite(output).all()


def test_clip_encoder_shapes():
    encoder = ClipEncoder(
        width=16,
        image_size=14,
        patch_size=7,
        depth=2,
        num_heads=2,
        hidden_dim=32,
        pre_layernorm_eps=1e-5,
        norm_eps=1e-5,
    )
    patch_embeds = torch.randn(2, 16, 2, 2)
    output = encoder(torch.randn(2, 3, 14, 14), patch_embeds)
    assert output.shape == (2, 5, 16)
    patch_embeds = torch.randn(2, 16, 3, 3)
    output = encoder(torch.randn(2, 3, 14, 14), patch_embeds)
    assert output.shape == (2, 10, 16)
    assert torch.isfinite(output).all()


def test_moe_routing_matches_manual_experts():
    torch.manual_seed(0)
    moe = MoE(
        hidden_size=16,
        moe_intermediate_size=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="softmax",
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
    )
    x = torch.randn(2, 5, 16)
    topk_idx, topk_weight = moe.gate(x)
    assert topk_idx.shape == (10, 2)
    assert topk_weight.shape == (10, 2)
    scores = moe.gate.weight.float() @ x.reshape(-1, 16).float().T
    expected_idx = scores.T.softmax(dim=-1).topk(2, dim=-1, sorted=False).indices
    for row in range(topk_idx.shape[0]):
        assert set(topk_idx[row].tolist()) == set(expected_idx[row].tolist())
    manual = moe.shared_experts(x).reshape(-1, 16)
    flat = x.reshape(-1, 16)
    rows = list(manual.unbind(0))
    for token in range(topk_idx.shape[0]):
        for slot in range(topk_idx.shape[1]):
            expert = moe.experts[int(topk_idx[token, slot])]
            rows[token] = rows[token] + (expert(flat[token : token + 1]) * topk_weight[token, slot]).reshape(16)
    manual = torch.stack(rows, dim=0).reshape(x.shape)
    torch.testing.assert_close(moe(x), manual, atol=1e-6, rtol=1e-6)


def test_page_aware_candidate_mask():
    input_ids = torch.tensor([[60, 61, 61, 60, 61, 61, 60, 62, 62, 62, 62, 62]])
    image_mask, image_pages, query_pages = build_page_aware_ids(input_ids, 61, 60, 2)
    assert image_mask is not None
    assert image_pages.tolist() == [[-1, 0, 0, -1, 1, 1, -1, -1, -1, -1, -1, -1]]
    assert query_pages.tolist() == [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    position_ids = torch.arange(12)[None]
    candidate = build_candidate_mask(position_ids[:, 6:12], 6, image_mask, image_pages, query_pages, 2)
    assert candidate.shape == (1, 1, 6, 6)
    assert candidate[0, 0, 7 - 6].tolist() == [True, True, False, False, False, False]
    candidate_image_query = build_candidate_mask(position_ids[:, 4:5], 6, image_mask, image_pages, query_pages, 2)
    assert candidate_image_query[0, 0, 0].tolist() == [False, False, True, False, False, False]


def test_page_rope_positions():
    input_ids = torch.tensor([[60, 61, 61, 60, 61, 61, 60, 62, 62, 62, 62, 62]])
    tok, page = build_page_rope_pos(input_ids, 61, 60)
    assert tok.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8]]
    assert page.tolist() == [[1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1]]


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


def test_page_rotary_2d_relative_invariance():
    rope = page_rope()
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    q_cos, q_sin = rope(page_positions(torch.tensor([[5]]), torch.tensor([[2]])))
    k_cos, k_sin = rope(page_positions(torch.tensor([[3]]), torch.tensor([[1]])))
    score = (apply_rope(q, q_cos, q_sin, dim=1) * apply_rope(k, k_cos, k_sin, dim=1)).sum(-1)
    q_cos, q_sin = rope(page_positions(torch.tensor([[8]]), torch.tensor([[4]])))
    k_cos, k_sin = rope(page_positions(torch.tensor([[6]]), torch.tensor([[3]])))
    shifted = (apply_rope(q, q_cos, q_sin, dim=1) * apply_rope(k, k_cos, k_sin, dim=1)).sum(-1)
    torch.testing.assert_close(score, shifted, atol=1e-5, rtol=1e-5)


def test_srswa_blocks_and_keep_mask():
    torch.manual_seed(0)
    router = BlockRouter(
        hidden_size=8,
        index_dim=8,
        eps=1e-6,
        block_size=2,
        local_blocks=1,
        topk_blocks=3,
        scaling_factor=1.0,
        sigma=1.0,
        prior_floor=1e-4,
        stage0="head_cls",
        query_chunk=8,
        image_only=True,
    )
    rope = page_rope()
    hidden = torch.randn(1, 12, 8)
    position_ids = torch.arange(12)[None]
    cos, sin = rope(page_positions(position_ids, torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]), mod=2))
    scores = router.block_scores(hidden, cos, sin, position_ids, 6)
    assert scores.shape == (1, 12, 6)
    assert torch.isfinite(scores[scores > -1e8]).all()
    candidate = torch.ones(1, 12, 6, dtype=torch.bool)
    top_indices, bias, valid, prior, _ = router.route_and_select(scores, candidate, None)
    assert top_indices.shape == (1, 12, 3)
    assert bias.shape == (1, 12, 3)
    assert valid.shape == (1, 12, 3)
    torch.testing.assert_close(prior.sum(dim=-1), torch.ones(1, 12), atol=1e-5, rtol=1e-5)
    assert valid[0, 11].sum() >= 1
    keep = sparse_keep_mask(top_indices, position_ids, 12, 2, 2, valid)
    assert keep.shape == (1, 1, 12, 12)
    causal = torch.arange(12)[None, :] <= position_ids[:, :, None]
    assert bool((keep & ~causal.unsqueeze(1)).any()) is False
    for query in range(12):
        assert bool(keep[0, 0, query, query])
        if query > 0:
            assert bool(keep[0, 0, query, query - 1])
        for slot in range(3):
            if not bool(valid[0, query, slot]):
                continue
            block = int(top_indices[0, query, slot])
            start = block * 2
            for key in range(start, min(start + 2, 12)):
                if key <= query:
                    assert bool(keep[0, 0, query, key])
        selected_tokens = set()
        for slot in range(3):
            if bool(valid[0, query, slot]):
                block = int(top_indices[0, query, slot])
                selected_tokens.update(range(block * 2, min(block * 2 + 2, 12)))
        for key in range(0, query):
            if key not in selected_tokens and key < query - 1:
                assert bool(keep[0, 0, query, key]) is False


def test_entmax15_normalized():
    x = torch.randn(2, 3, 7)
    out = entmax15(x, dim=-1)
    assert out.shape == x.shape
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(2, 3), atol=1e-5, rtol=1e-5)
    assert bool((out >= 0).all())


def test_decoder_layer_and_model_shapes():
    torch.manual_seed(0)
    config = tiny_config()
    model = UnlimitedOCRModel(config)
    input_ids = tiny_page_ids()
    pixel_values = torch.randn(2, 3, 56, 56)
    output = model(input_ids=input_ids, pixel_values=pixel_values)
    assert output.shape == (1, input_ids.shape[1], config.hidden_size)
    assert torch.isfinite(output).all()
    logits = UnlimitedOCR(config)(input_ids=input_ids, pixel_values=pixel_values)
    assert logits.shape == (1, input_ids.shape[1], config.vocab_size)
    assert torch.isfinite(logits).all()


def test_upstream_deepencoder_reference(reference_path):
    if reference_path is None or not os.path.isdir(reference_path):
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
    spec = importlib.util.spec_from_file_location("upstream_deepencoder", {os.path.join(reference_path, "deepencoder.py")!r})
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
except Exception as exc:
    print("UPSTREAM_IMPORT_ERROR", repr(exc))
    raise SystemExit(2)

from nnm.models.unlimited_ocr.sam import SamEncoder
from nnm.models.unlimited_ocr.clip import ClipEncoder

torch.manual_seed(0)
up_sam = upstream.ImageEncoderViT(
    depth=2, embed_dim=16, img_size=56, mlp_ratio=1,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=2, patch_size=7,
    qkv_bias=True, use_rel_pos=True, global_attn_indexes=[1], window_size=4, out_chans=256,
)
our_sam = SamEncoder(
    image_size=56, patch_size=7, width=16, depth=2, num_heads=2, mlp_ratio=1,
    out_chans=256, window_size=4, global_attn_indexes=[1],
    downsample_channels=(512, 1024), norm_eps=1e-6,
)
up_sam.load_state_dict(our_sam.state_dict())
pixel_values = torch.randn(2, 3, 56, 56)
reference_sam = up_sam(pixel_values)
candidate_sam = our_sam(pixel_values)
torch.testing.assert_close(reference_sam, candidate_sam, atol=1e-4, rtol=1e-4)

clip_cfg = upstream.adict(
    num_layers=2, hidden_size=16, num_heads=2, num_attention_heads=2, ffn_hidden_size=32,
    seq_length=4, max_position_embeddings=4, use_flash_attn=False,
    understand_projector_stride=2, hidden_dropout=0.0, attention_dropout=0.0,
    no_persist_layer_norm=False, layernorm_epsilon=1e-5, pre_layernorm_epsilon=1e-5,
    image_size=14, patch_size=7, recompute_list=[],
)
up_clip = upstream.VitModel(cfg=clip_cfg, freeze_embed=False, freeze_pre_norm=False)
our_clip = ClipEncoder(
    width=16, image_size=14, patch_size=7, depth=2, num_heads=2, hidden_dim=32,
    pre_layernorm_eps=1e-5, norm_eps=1e-5,
)
up_clip.load_state_dict(our_clip.state_dict())
patch_embeds = torch.randn(2, 16, 2, 2)
torch.testing.assert_close(up_clip(pixel_values, patch_embeds), our_clip(pixel_values, patch_embeds), atol=1e-4, rtol=1e-4)
patch_embeds = torch.randn(2, 16, 3, 3)
torch.testing.assert_close(up_clip(pixel_values, patch_embeds), our_clip(pixel_values, patch_embeds), atol=1e-4, rtol=1e-4)
print("UPSTREAM_OK")
"""
    result = subprocess.run(
        [upstream_python, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    if result.returncode == 2:
        pytest.skip(f"upstream dynamic code cannot run: {result.stdout.strip()} {result.stderr.strip()}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UPSTREAM_OK" in result.stdout
