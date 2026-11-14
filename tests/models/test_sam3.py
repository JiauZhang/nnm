import re

import torch
from conippets.config import Config
from transformers import Sam3ViTConfig, Sam3ViTModel

from nnm.models.sam3 import SAM3, SAM31, Sam3ViT


def vision_config(**overrides):
    values = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=32,
        patch_size=8,
        layer_norm_eps=1e-6,
        rope_theta=10000.0,
        window_size=2,
        global_attn_indexes=[1],
        pretrain_image_size=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return values


def tiny_config():
    return Config(
        vision_config=dict(
            backbone_config=vision_config(global_attn_indexes=[0, 1]),
            fpn_hidden_size=32,
            scale_factors=[4.0, 2.0, 1.0, 0.5],
            layer_norm_eps=1e-6,
        ),
        text_config=dict(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=32,
            layer_norm_eps=1e-5,
        ),
        geometry_encoder_config=dict(
            hidden_size=32,
            num_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            dropout=0.0,
            hidden_act="relu",
            roi_size=2,
            layer_norm_eps=1e-6,
        ),
        detr_encoder_config=dict(
            hidden_size=32,
            num_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            dropout=0.0,
            hidden_act="relu",
            layer_norm_eps=1e-6,
        ),
        detr_decoder_config=dict(
            hidden_size=32,
            num_layers=1,
            num_queries=4,
            num_attention_heads=4,
            intermediate_size=64,
            dropout=0.0,
            hidden_act="relu",
            layer_norm_eps=1e-6,
        ),
        mask_decoder_config=dict(
            hidden_size=32,
            num_upsampling_stages=3,
            num_attention_heads=4,
            dropout=0.0,
            layer_norm_eps=1e-6,
        ),
    )


def test_sam3_vision_matches_transformers():
    config = Sam3ViTConfig(**vision_config(global_attn_indexes=[0, 1]))
    nnm_config = Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=32,
        patch_size=8,
        layer_norm_eps=1e-6,
        rope_theta=10000.0,
        window_size=2,
        global_attn_indexes=[0, 1],
        pretrain_image_size=32,
        pretrain_use_cls_token=False,
        hidden_dropout=0.0,
    )
    torch.manual_seed(0)
    reference = Sam3ViTModel(config).eval()
    model = Sam3ViT(nnm_config).eval()
    state_dict = {}
    for key, value in reference.state_dict().items():
        key = key.replace("embeddings.patch_embeddings.projection.", "embeddings.projection.")
        key = key.replace("embeddings.position_embeddings", "embeddings.pos_embed")
        key = key.replace("layer_norm1", "norm1").replace("layer_norm2", "norm2")
        key = key.replace("attention.", "attn.")
        key = re.sub(r"layers\.(\d+)\.", r"layers.\1.block.", key)
        key = key.replace("layer_norm", "ln_pre")
        state_dict[key] = value
    model.load_state_dict(state_dict, strict=True)
    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = reference(pixel_values).last_hidden_state
        actual = model(pixel_values).reshape(2, 16, -1)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_sam3_forward_with_text():
    torch.manual_seed(0)
    model = SAM3(tiny_config()).eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 100, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(pixel_values, input_ids, attention_mask)
    assert output["pred_masks"].shape == (2, 4, 16, 16)
    assert output["pred_logits"].shape == (2, 4)
    assert output["pred_boxes"].shape == (2, 4, 4)
    assert output["semantic_seg"].shape == (2, 1, 16, 16)


def test_sam31_memory_and_multiplex():
    config = tiny_config()
    config.memory_dim = 32
    config.memory_layers = 1
    config.memory_attention_layers = 1
    config.multiplex_count = 2
    config.num_multimask_outputs = 1
    model = SAM31(config).eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    masks = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        output = model.forward_video(pixel_values, masks)
    assert output["masks"].shape == (2, 2, 2, 16, 16)
    assert output["iou_pred"].shape == (2, 2, 2)
    assert output["object_score_logits"].shape == (2, 2, 1)


def test_sam3_forward_with_boxes():
    torch.manual_seed(0)
    model = SAM3(tiny_config()).eval()
    pixel_values = torch.randn(1, 3, 32, 32)
    input_ids = torch.randint(0, 100, (1, 5))
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])
    labels = torch.ones(1, 1, dtype=torch.long)
    with torch.no_grad():
        output = model(pixel_values, input_ids, input_boxes=boxes, input_boxes_labels=labels)
    assert output["pred_masks"].shape == (1, 4, 16, 16)
