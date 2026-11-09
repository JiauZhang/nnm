import torch
from conippets.config import Config

from nnm.models.audio8_tts import Audio8TTS


def tiny_config():
    return Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        norm_eps=1e-5,
        hidden_act="silu",
        rope_theta=10000.0,
        rope_base=10000.0,
        pad_token_id=0,
        eos_token_id=127,
        embedding_multiplier=0.1,
        mamba_d_ssm=64,
        mamba_d_head=16,
        mamba_d_state=16,
        mamba_n_heads=4,
        mamba_n_groups=1,
        mamba_d_conv=4,
        mamba_chunk_size=8,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        projectors_bias=False,
        mlp_bias=False,
        n_fast_layer=2,
        fast_dim=32,
        fast_n_head=4,
        fast_n_local_heads=2,
        fast_head_dim=8,
        fast_intermediate_size=64,
        fast_attention_qkv_bias=False,
        fast_attention_o_bias=False,
        codebook_size=32,
        num_codebooks=3,
        semantic_begin_id=100,
        semantic_end_id=110,
    )


def test_audio8_tts_forward():
    torch.manual_seed(0)
    config = tiny_config()
    model = Audio8TTS(config).eval()
    batch, length = 2, 5
    semantic = torch.randint(config.semantic_begin_id, config.semantic_end_id + 1, (batch, length))
    codebooks = torch.randint(0, config.codebook_size, (batch, config.num_codebooks, length))
    input_ids = torch.cat([semantic[:, None], codebooks], dim=1)
    attn_mask = torch.ones(batch, length, dtype=torch.long)
    with torch.no_grad():
        logits, hidden = model(input_ids, attn_mask=attn_mask)
        repeated, _ = model(input_ids, attn_mask=attn_mask)
    assert logits.shape == (batch, length, config.codebook_size + 1)
    assert hidden.shape == (batch, length, config.hidden_size)
    assert torch.equal(logits, repeated)
