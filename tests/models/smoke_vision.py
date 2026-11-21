import torch

from tests.models.smoke_registry import SmokeCase, make, shape


def check_davit(model, output):
    assert tuple(output.shape) == (2, 32, 8, 8)
    _, features = model.forward_features(torch.randn(1, 3, 64, 64))
    assert len(features) == 2
    assert tuple(features[0].shape) == (1, 16, 16, 16)
    assert tuple(features[1].shape) == (1, 32, 8, 8)


def check_siglip2(model, output):
    assert tuple(output["logits_per_image"].shape) == (2, 2)
    assert tuple(output["logits_per_text"].shape) == (2, 2)
    vision = model.vision_model(torch.randn(1, 3, 32, 32))
    assert tuple(vision["last_hidden_state"].shape) == (1, 16, 32)
    assert tuple(vision["pooler_output"].shape) == (1, 32)


def check_florence2(model, output):
    assert tuple(output.shape) == (2, 6, 64)
    features = model.get_image_features(torch.randn(1, 3, 32, 32))
    assert tuple(features.shape) == (1, 17, 16)


def check_fastvit(model, output):
    assert tuple(output.shape) == (1, 10)
    features = model.forward_features(torch.randn(1, 3, 64, 64))
    assert tuple(features.shape) == (1, 64, 8, 8)


def check_perception_encoder(model, output):
    assert tuple(output["x_norm_patchtokens"].shape) == (2, 16, 32)
    assert tuple(output["x_storage_tokens"].shape) == (2, 2, 32)
    assert tuple(output["logits_per_image"].shape) == (2, 2)
    vision = model.vision_model.forward_features(torch.randn(1, 3, 32, 32))
    assert tuple(vision["x_norm_clstoken"].shape) == (1, 32)


def check_depthanything2(model, output):
    assert tuple(output.shape) == (1, 32, 32)


def check_depthpro2(model, output):
    assert tuple(output.shape) == (2, 32, 32)


CASES = [
    SmokeCase(
        "davit",
        "dual-branch",
        make(
            "nnm.models.davit",
            "DaViT",
            image_size=64,
            num_channels=3,
            embed_dims=[16, 32],
            depths=[1, 1],
            num_heads=[1, 2],
            num_groups=[1, 2],
            patch_size=[7, 3],
            patch_stride=[4, 2],
            patch_padding=[3, 1],
            patch_prenorm=[False, True],
            window_size=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
        ),
        lambda: (torch.randn(2, 3, 64, 64),),
        check_davit,
    ),
    SmokeCase(
        "siglip2",
        "dual",
        make(
            "nnm.models.siglip2",
            "SigLIP2",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=32,
            patch_size=8,
            layer_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            vision_use_head=True,
            vocab_size=100,
            max_position_embeddings=16,
        ),
        lambda: (
            torch.randn(2, 3, 32, 32),
            torch.randint(0, 100, (2, 8)),
            torch.ones(2, 8, dtype=torch.long),
        ),
        check_siglip2,
    ),
    SmokeCase(
        "florence2",
        "encoder-decoder",
        make(
            "nnm.models.florence",
            "Florence2",
            image_token_id=5,
            vision_config=dict(
                image_size=32,
                in_channels=3,
                num_channels=3,
                embed_dim=[8, 16],
                embed_dims=[8, 16],
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
        ),
        lambda: (),
        check_florence2,
        call=lambda model: model(
            torch.randn(2, 3, 32, 32),
            torch.full((2, 17), 5),
            decoder_input_ids=torch.randint(0, 64, (2, 6)),
        ),
    ),
    SmokeCase(
        "fastvit",
        "repmixer",
        make(
            "nnm.models.fastvit",
            "FastViT",
            num_channels=3,
            depths=[2, 2],
            embed_dims=[16, 32],
            token_mixers=["repmixer", "repmixer"],
            mlp_ratios=[4.0, 4.0],
            downsamples=[False, True],
            num_classes=10,
            drop_path_rate=0.0,
            layer_scale_init_value=1e-5,
        ),
        lambda: (torch.randn(1, 3, 64, 64),),
        check_fastvit,
    ),
    SmokeCase(
        "perception-encoder",
        "dual",
        make(
            "nnm.models.perception_encoder",
            "PerceptionEncoder",
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=8,
            num_channels=3,
            num_register_tokens=2,
            layer_norm_eps=1e-6,
            layerscale_init=1e-5,
            rope_theta=100.0,
            pos_embed_rope_normalize_coords="separate",
            use_text=True,
            text_hidden_size=32,
            text_num_attention_heads=4,
            text_num_hidden_layers=2,
            text_intermediate_size=64,
            text_vocab_size=100,
            text_rope_base=10000.0,
            projection_dim=32,
        ),
        lambda: (
            torch.randn(2, 3, 32, 32),
            torch.randint(0, 100, (2, 8)),
            torch.ones(2, 8, dtype=torch.long),
        ),
        check_perception_encoder,
    ),
    SmokeCase(
        "depthpro2",
        "monocular-depth",
        make(
            "nnm.models.depthpro2",
            "DepthPro2",
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=2,
            layer_norm_eps=1e-6,
            patch_size=8,
            rope_theta=100.0,
            pos_embed_rope_normalize_coords="max",
            layerscale_init=1e-5,
            qkv_bias=False,
            proj_bias=True,
            hidden_act="gelu",
            fov=False,
            reassembled_blocks=[0, 1, 2, 3],
            cross_frame_blocks=[1, 3],
            dims_reassemble=[8, 16, 32, 32],
            dim_decoder=16,
        ),
        lambda: (torch.randn(2, 3, 32, 32),),
        check_depthpro2,
        call=lambda model: model.predict_inverse_depth(torch.randn(2, 3, 32, 32)),
    ),
    SmokeCase(
        "depthpro2-fov",
        "focal-estimator",
        make(
            "nnm.models.depthpro2",
            "FOVEstimator",
            hidden_size=32,
            num_hidden_layers=12,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=2,
            layer_norm_eps=1e-6,
            patch_size=8,
            rope_theta=100.0,
            pos_embed_rope_normalize_coords="max",
            layerscale_init=1e-5,
            qkv_bias=False,
            proj_bias=True,
            hidden_act="gelu",
        ),
        lambda: (torch.randn(1, 3, 64, 64),),
        shape(1, 2),
    ),
    SmokeCase(
        "depthanything2",
        "monocular-depth",
        make(
            "nnm.models.depthanything2",
            "DepthAnythingV2",
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=64,
            num_channels=3,
            patch_size=8,
            image_size=32,
            layer_norm_eps=1e-6,
            layerscale_init=1e-5,
            num_register_tokens=0,
            qkv_bias=True,
            proj_bias=True,
            hidden_act="gelu",
            features=16,
            out_channels=[8, 16, 32, 32],
            use_clstoken=False,
            tapped_blocks=[0, 1, 2, 3],
            variant="vits",
        ),
        lambda: (torch.randn(1, 3, 32, 32),),
        check_depthanything2,
    ),
]
