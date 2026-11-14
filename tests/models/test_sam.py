import torch
from conippets.config import Config
from transformers import SamConfig, SamMaskDecoderConfig, SamModel, SamPromptEncoderConfig, SamVisionConfig

from nnm.models.sam import SAM


VISION = dict(
    hidden_size=32,
    output_channels=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_channels=3,
    image_size=32,
    patch_size=8,
    hidden_act="gelu",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
    qkv_bias=True,
    mlp_ratio=4.0,
    use_abs_pos=True,
    use_rel_pos=True,
    window_size=4,
    global_attn_indexes=[1],
    num_pos_feats=16,
    mlp_dim=64,
)

PROMPT = dict(
    hidden_size=32,
    image_size=32,
    patch_size=8,
    mask_input_channels=16,
    num_point_embeddings=4,
    hidden_act="gelu",
    layer_norm_eps=1e-6,
)

MASK = dict(
    hidden_size=32,
    hidden_act="relu",
    mlp_dim=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    attention_downsample_rate=2,
    num_multimask_outputs=3,
    iou_head_depth=3,
    iou_head_hidden_dim=32,
    layer_norm_eps=1e-6,
)


def config():
    return Config(vision_config=VISION, prompt_encoder_config=PROMPT, mask_decoder_config=MASK)


def reference_config():
    vision_config = SamVisionConfig(**VISION)
    prompt_encoder_config = SamPromptEncoderConfig(**PROMPT)
    mask_decoder_config = SamMaskDecoderConfig(**MASK)
    config = SamConfig(
        vision_config=vision_config,
        prompt_encoder_config=prompt_encoder_config,
        mask_decoder_config=mask_decoder_config,
    )
    config._attn_implementation = "eager"
    return config


def map_state_dict(state_dict):
    mapped = {}
    for key, value in state_dict.items():
        key = key.replace("vision_encoder.patch_embed.projection.", "vision_encoder.patch_embed.proj.")
        key = key.replace(".mlp.lin1.", ".mlp.fc1.").replace(".mlp.lin2.", ".mlp.fc2.")
        for index in range(VISION["num_hidden_layers"]):
            key = key.replace(f"vision_encoder.layers.{index}.layer_norm1.", f"vision_encoder.layers.{index}.norm1.")
            key = key.replace(f"vision_encoder.layers.{index}.layer_norm2.", f"vision_encoder.layers.{index}.norm2.")
        mapped[key] = value
    return mapped


def build_models():
    torch.manual_seed(0)
    reference = SamModel(reference_config()).eval()
    state_dict = {key: value + 0.02 * torch.randn_like(value) for key, value in reference.state_dict().items()}
    shared = state_dict["shared_image_embedding.positional_embedding"]
    state_dict["prompt_encoder.shared_embedding.positional_embedding"] = shared.clone()
    reference.load_state_dict(state_dict, strict=True)
    model = SAM(config()).eval()
    model.load_state_dict(map_state_dict(state_dict), strict=True)
    return reference, model


def inputs(batch_size=2):
    generator = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(batch_size, 3, 32, 32, generator=generator)
    input_points = torch.rand(batch_size, 1, 2, 2, generator=generator) * 32
    input_labels = torch.ones(batch_size, 1, 2, dtype=torch.long)
    input_labels[:, :, 1] = 0
    input_boxes = torch.rand(batch_size, 1, 4, generator=generator) * 32
    input_masks = torch.randn(batch_size, 1, 16, 16, generator=generator)
    return pixel_values, input_points, input_labels, input_boxes, input_masks


def test_vision_encoder_matches_transformers():
    reference, model = build_models()
    pixel_values, *_ = inputs()
    with torch.no_grad():
        expected = reference.vision_encoder(pixel_values).last_hidden_state
        actual = model.vision_encoder(pixel_values)
    assert actual.shape == (2, 32, 4, 4)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_prompt_encoder_matches_transformers():
    reference, model = build_models()
    _, input_points, input_labels, input_boxes, input_masks = inputs()
    with torch.no_grad():
        expected = reference.prompt_encoder(input_points, input_labels, input_boxes, input_masks)
        actual = model.prompt_encoder(input_points, input_labels, input_boxes, input_masks)
        expected_dense_pe = reference.get_image_wide_positional_embeddings()
        actual_dense_pe = model.get_dense_pe()
    torch.testing.assert_close(actual[0], expected[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_dense_pe, expected_dense_pe, atol=1e-5, rtol=1e-5)


def test_prompt_encoder_padding_matches_transformers():
    reference, model = build_models()
    _, input_points, input_labels, _, _ = inputs()
    with torch.no_grad():
        expected = reference.prompt_encoder(input_points, input_labels, None, None)
        actual = model.prompt_encoder(input_points, input_labels, None, None)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-5, rtol=1e-5)


def test_mask_decoder_matches_transformers():
    reference, model = build_models()
    pixel_values, input_points, input_labels, input_boxes, input_masks = inputs()
    with torch.no_grad():
        image_embeddings = reference.vision_encoder(pixel_values).last_hidden_state
        image_pe = reference.get_image_wide_positional_embeddings()
        sparse, dense = reference.prompt_encoder(input_points, input_labels, input_boxes, input_masks)
        expected = reference.mask_decoder(image_embeddings, image_pe, sparse, dense, multimask_output=True)
        actual = model.mask_decoder.predict_masks(image_embeddings, image_pe, sparse, dense)
    torch.testing.assert_close(actual[0][:, :, 1:], expected[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1][:, :, 1:], expected[1], atol=1e-5, rtol=1e-5)


def test_sam_forward_matches_transformers():
    reference, model = build_models()
    pixel_values, input_points, input_labels, input_boxes, input_masks = inputs()
    with torch.no_grad():
        expected = reference(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
            multimask_output=True,
        )
        actual = model(
            pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
    assert actual["pred_masks"].shape == (2, 1, 3, 16, 16)
    assert actual["iou_scores"].shape == (2, 1, 3)
    torch.testing.assert_close(actual["pred_masks"], expected.pred_masks, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual["iou_scores"], expected.iou_scores, atol=1e-5, rtol=1e-5)


def test_sam_forward_points_only():
    reference, model = build_models()
    pixel_values, input_points, input_labels, _, _ = inputs()
    with torch.no_grad():
        expected = reference(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            multimask_output=False,
        )
        actual = model(pixel_values, input_points=input_points, input_labels=input_labels, multimask_output=False)
    assert actual["pred_masks"].shape == (2, 1, 1, 16, 16)
    assert actual["iou_scores"].shape == (2, 1, 1)
    torch.testing.assert_close(actual["pred_masks"], expected.pred_masks, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual["iou_scores"], expected.iou_scores, atol=1e-5, rtol=1e-5)
