import pytest
import torch

from conippets.config import Config
from nnm.models.oxr import OXR
from nnm.models.oxr.qwen3 import Qwen3ForCausalLM
from nnm.models.oxr.vit import Siglip2NaViT


def vit_config():
    return Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        patch_size=4,
        image_size=32,
        hidden_stride=2,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        use_rope=True,
        window_size=16,
        preserve_original_pe=True,
        fullatt_block_indexes=None,
        num_patches=-1,
        temporal_patch_size=1,
    )


def qwen_config(use_cache=False):
    return Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1000000,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        pad_token_id=0,
        use_cache=use_cache,
    )


@pytest.fixture(scope="module")
def oxr_config(weight_config):
    return weight_config("OXR-1.0")


class TestOxr:
    def test_config_drives_the_architecture(self, oxr_config):
        model = OXR(oxr_config)
        assert sum(parameter.numel() for parameter in model.parameters()) == 1378795752

    def test_vision_encoder_preserves_the_patch_count(self):
        model = Siglip2NaViT(vit_config()).eval()
        pixel_values = torch.randn(16, 48)
        grid = torch.tensor([[1, 4, 4]])
        with torch.no_grad():
            last_hidden_state, hidden_states = model(pixel_values, grid, output_hidden_states=True)

        assert last_hidden_state.shape == (16, 32)
        assert len(hidden_states) == 2

    def test_qwen3_matches_transformers(self):
        transformers = pytest.importorskip("transformers")
        config = qwen_config()
        model = Qwen3ForCausalLM(config).eval()
        reference_config = transformers.Qwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=True,
            attention_bias=False,
        )
        reference = transformers.Qwen3ForCausalLM(reference_config).to(torch.float32).eval()
        state = {
            key.replace("model.", "backbone.", 1): value
            for key, value in reference.state_dict().items()
            if not key.startswith("lm_head.")
        }
        model.load_state_dict(state, strict=False)

        input_ids = torch.randint(0, 100, (2, 8))
        embeddings = reference.get_input_embeddings()(input_ids)
        with torch.no_grad():
            expected = reference(inputs_embeds=embeddings).logits
            actual = model(embeddings)

        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_soft_visual_tokens_are_weighted_embeddings(self, oxr_config):
        model = OXR(oxr_config)
        tokens = torch.zeros(1, model.config.visual_vocab_size)
        tokens[0, 0] = 1.0
        assert torch.equal(model.visual_embed(tokens), model.vte.weight[0].unsqueeze(0))

    def test_merge_multimodal_replaces_visual_placeholders(self, oxr_config):
        model = OXR(oxr_config)
        input_ids = torch.tensor([[5, -301, -300]])
        visual_tokens = torch.zeros(1, model.config.visual_vocab_size)
        visual_tokens[0, 0] = 1.0
        with torch.no_grad():
            embeddings = model.merge_multimodal(input_ids, visual_tokens=visual_tokens)
        assert embeddings.shape == (1, 3, model.config.hidden_size)
        assert torch.equal(embeddings[:, 2], model.vte.weight[0].expand(1, -1))
