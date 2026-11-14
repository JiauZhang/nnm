import torch

from tests.models.smoke_registry import SmokeCase, from_config, make, shape
from tests.models.test_d1 import vl_config as d1_vl_config
from tests.models.test_d1 import vl_inputs as d1_vl_inputs
from tests.models.test_lfm3_audio import tiny_config as audio_config
from tests.models.test_lfm3_audio import tiny_inputs as audio_inputs
from tests.models.test_lfm3_omni import tiny_config as omni_config
from tests.models.test_lfm3_omni import tiny_inputs as omni_inputs
from tests.models.test_lfm3_vl import GRID, IMAGE_TOKENS, TINY, NUM_PATCHES, pixel_values_for
from tests.models.test_lfm3_vl import tiny_config as vl_config_lfm3


def check_bart(model, output):
    assert tuple(output.shape) == (1, 5, 64)
    generated = model.generate(torch.randint(4, 64, (1, 8)), max_length=16)
    assert generated.shape[0] == 1
    assert torch.isfinite(output).all()


def check_lfm3_vl(model, output):
    pixel_values = pixel_values_for(GRID)
    input_ids = torch.randint(0, TINY["image_token_id"], (2, 18))
    input_ids[0, 5 : 5 + IMAGE_TOKENS[0]] = TINY["image_token_id"]
    input_ids[1, 5 : 5 + IMAGE_TOKENS[1]] = TINY["image_token_id"]
    hidden_states = model.vision_tower(pixel_values, GRID)
    merged = model.multi_modal_projector(hidden_states)
    features = model.get_image_features(pixel_values, GRID)
    assert tuple(hidden_states.shape) == (NUM_PATCHES, 32)
    assert tuple(merged.shape) == (sum(IMAGE_TOKENS), 32)
    assert [feature.shape[0] for feature in features] == IMAGE_TOKENS
    assert tuple(output.shape) == (2, 18, 100)


def check_lfm3_omni(model, output):
    input_ids, input_features, feature_mask, codes, pixel_values, image_grid = omni_inputs()
    audio = model.get_audio_features(input_features, feature_mask)
    image = model.get_image_features(pixel_values, image_grid)
    assert tuple(audio.shape) == (10, 64)
    assert [tuple(feature.shape) for feature in image] == [(4, 64)]
    assert tuple(output.shape) == (1, 24, 100)


def check_lfm3_audio(model, output):
    input_ids, input_features, feature_mask, codes = audio_inputs()
    features = model.get_audio_features(input_features, feature_mask)
    out_embeds = model.get_audio_out_embeds(codes)
    depth = model.audio.depth_decoder(torch.randn(2, 64), codes, torch.zeros(2, 3, dtype=torch.bool))
    assert tuple(features.shape) == (14, 64)
    assert tuple(out_embeds.shape) == (2, 64)
    assert tuple(depth.shape) == (2, 3, 11)
    assert tuple(output.shape) == (2, 16, 100)


def vl_inputs():
    pixel_values = pixel_values_for(GRID)
    input_ids = torch.randint(0, TINY["image_token_id"], (2, 18))
    input_ids[0, 5 : 5 + IMAGE_TOKENS[0]] = TINY["image_token_id"]
    input_ids[1, 5 : 5 + IMAGE_TOKENS[1]] = TINY["image_token_id"]
    return input_ids, pixel_values, GRID


CASES = [
    SmokeCase(
        "lfm3-vl",
        "dual",
        from_config("nnm.models.lfm3", "Lfm3VlModel", vl_config_lfm3),
        lambda: (),
        check_lfm3_vl,
        call=lambda model: model(*vl_inputs()),
    ),
    SmokeCase(
        "lfm3-omni",
        "audio-vision-language",
        from_config("nnm.models.lfm3", "Lfm3OmniModel", omni_config),
        lambda: (),
        check_lfm3_omni,
        call=lambda model: model(*omni_inputs()),
    ),
    SmokeCase(
        "lfm3-audio",
        "duplex",
        from_config("nnm.models.lfm3", "Lfm3AudioModel", audio_config),
        lambda: (),
        check_lfm3_audio,
        call=lambda model: model(*audio_inputs()),
    ),
]

CASES.extend(
    [
        SmokeCase(
            "d1",
            "vl",
            from_config("nnm.models.lfm3", "D1VlModel", d1_vl_config),
            lambda: (),
            shape(2, 12, 100),
            call=lambda model: model(*d1_vl_inputs()),
        ),
        SmokeCase(
            "d1",
            "omni",
            from_config("nnm.models.lfm3", "D1OmniModel", omni_config),
            lambda: (),
            shape(1, 24, 100),
            call=lambda model: model(*omni_inputs()),
        ),
    ]
)

CASES.append(
    SmokeCase(
        "bart",
        "encoder-decoder",
        make(
            "nnm.models.bart",
            "BartForConditionalGeneration",
            vocab_size=64,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
            forced_eos_token_id=2,
            activation_function="gelu",
            dropout=0.0,
            activation_dropout=0.0,
            tie_word_embeddings=True,
        ),
        lambda: (),
        check_bart,
        call=lambda model: model(torch.randint(4, 64, (1, 8)), torch.randint(4, 64, (1, 5))),
    )
)
