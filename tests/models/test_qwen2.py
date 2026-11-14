import torch
import pytest
import random
from torch import nn
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from transformers.models.qwen2 import configuration_qwen2 as cfg
from nnm.cache import Cache, KVCache
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.attention import RotarySelfAttention
from nnm.models.qwen2 import (
    Qwen2MLP,
    Qwen2DecoderLayer,
    Qwen2Model,
)

parametrize = pytest.mark.parametrize


def init_linear_weight(src, dst, bias=True):
    assert dst.weight.shape == src.weight.shape
    dst.weight = src.weight
    if bias:
        assert dst.bias.shape == src.bias.shape
        dst.bias = src.bias


def init_linear_weight_kv(nnm_k_proj, nnm_v_proj, hf_k_proj, hf_v_proj):
    assert nnm_k_proj.weight.shape == hf_k_proj.weight.shape
    assert nnm_k_proj.bias.shape == hf_k_proj.bias.shape
    nnm_k_proj.weight = torch.nn.Parameter(hf_k_proj.weight.clone())
    nnm_k_proj.bias = torch.nn.Parameter(hf_k_proj.bias.clone())

    assert nnm_v_proj.weight.shape == hf_v_proj.weight.shape
    assert nnm_v_proj.bias.shape == hf_v_proj.bias.shape
    nnm_v_proj.weight = torch.nn.Parameter(hf_v_proj.weight.clone())
    nnm_v_proj.bias = torch.nn.Parameter(hf_v_proj.bias.clone())


@parametrize("num_kv_heads", [4, 8])
@parametrize("embed_dim", [32, 64])
@parametrize("batch", [1, 2])
@torch.no_grad()
def test_qwen2_attn(batch, embed_dim, num_kv_heads):
    num_attn_heads = 8
    max_seq_len, seq_len, base = 512, 234, 2333
    x = torch.randn(batch, seq_len, embed_dim)
    position_ids = torch.arange(seq_len)
    head_dim = embed_dim // num_attn_heads

    rope = RoPE(rotary_inv_freq(head_dim, base))
    nnm_cos, nnm_sin = rope(position_ids)
    nnm_attn = RotarySelfAttention(
        embed_dim,
        num_attn_heads,
        num_kv_heads,
        qkv_bias=True,
        o_bias=False,
    )

    config = cfg.Qwen2Config(
        hidden_size=embed_dim,
        num_attention_heads=num_attn_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rope_theta=base,
        max_position_embeddings=max_seq_len,
    )
    config._attn_implementation = "sdpa"
    hf_attn = qwen2.Qwen2Attention(config, 0)

    init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj)
    nnm_q = nnm_attn.q_proj(x)
    hf_q = hf_attn.q_proj(x)
    assert nnm_q.shape == hf_q.shape
    torch.testing.assert_close(nnm_q, hf_q, atol=1e-5, rtol=1e-5)

    init_linear_weight(nnm_attn.o_proj, hf_attn.o_proj, bias=False)
    nnm_o = nnm_attn.o_proj(x)
    hf_o = hf_attn.o_proj(x)
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)

    init_linear_weight_kv(nnm_attn.k_proj, nnm_attn.v_proj, hf_attn.k_proj, hf_attn.v_proj)
    nnm_k, nnm_v = nnm_attn.k_proj(x), nnm_attn.v_proj(x)
    hf_k, hf_v = hf_attn.k_proj(x), hf_attn.v_proj(x)
    assert nnm_k.shape == hf_k.shape and nnm_v.shape == hf_v.shape
    torch.testing.assert_close(nnm_k, hf_k, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(nnm_v, hf_v, atol=1e-5, rtol=1e-5)

    hf_rope = qwen2.Qwen2RotaryEmbedding(config=config)
    position_ids = position_ids.unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)

    nnm_sin = nnm_sin.unsqueeze(0)
    nnm_cos = nnm_cos.unsqueeze(0)

    assert nnm_cos.shape == hf_cos.shape and nnm_sin.shape == hf_sin.shape
    torch.testing.assert_close(nnm_cos, hf_cos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(nnm_sin, hf_sin, atol=1e-5, rtol=1e-5)

    nnm_o = nnm_attn(x, rotary_embeddings=(nnm_cos, nnm_sin))
    hf_attn.eval()

    hf_o, hf_attn_weight = hf_attn(x, (hf_cos, hf_sin), None)
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


@parametrize("eps", [1e-6, 1e-8])
@parametrize("embed_dim", [64, 128])
@parametrize("seq_len", [123, 233])
@parametrize("batch", [1, 2])
@torch.no_grad()
def test_qwen2_rms_norm(batch, seq_len, embed_dim, eps):
    hf_norm = qwen2.Qwen2RMSNorm(embed_dim, eps=eps)
    nnm_norm = nn.RMSNorm(embed_dim, eps=eps, elementwise_affine=True)
    assert nnm_norm.weight.shape == hf_norm.weight.shape

    nnm_norm.weight = torch.nn.Parameter(torch.randn(*nnm_norm.weight.shape))
    x = torch.randn(batch, seq_len, embed_dim)
    hf_norm.weight = nnm_norm.weight
    nnm_o = nnm_norm(x)
    hf_o = hf_norm(x)
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


def init_mlp_weight(src, dst):
    init_linear_weight(src.gate_proj, dst.gate_proj, bias=False)
    init_linear_weight(src.up_proj, dst.up_proj, bias=False)
    init_linear_weight(src.down_proj, dst.down_proj, bias=False)


@parametrize("intermediate_size", [256, 384])
@parametrize("embed_dim", [64, 128])
@parametrize("seq_len", [123, 233])
@parametrize("batch", [1, 2])
@torch.no_grad()
def test_qwen2_mlp(batch, seq_len, embed_dim, intermediate_size):
    config = cfg.Qwen2Config(hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu")
    hf_mlp = qwen2.Qwen2MLP(config)
    nnm_mlp = Qwen2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size)

    init_mlp_weight(nnm_mlp, hf_mlp)
    x = torch.randn(batch, seq_len, embed_dim)
    nnm_o = nnm_mlp(x)
    hf_o = hf_mlp(x)
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


def init_decoder_layer(nnm_decoder, hf_decoder):
    init_mlp_weight(nnm_decoder.mlp, hf_decoder.mlp)
    init_linear_weight(nnm_decoder.input_layernorm, hf_decoder.input_layernorm, bias=False)
    init_linear_weight(nnm_decoder.post_attention_layernorm, hf_decoder.post_attention_layernorm, bias=False)
    nnm_attn, hf_attn = nnm_decoder.self_attn, hf_decoder.self_attn
    init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj)
    init_linear_weight(nnm_attn.o_proj, hf_attn.o_proj, bias=False)
    init_linear_weight_kv(nnm_attn.k_proj, nnm_attn.v_proj, hf_attn.k_proj, hf_attn.v_proj)
    hf_decoder.eval()


@parametrize("num_kv_heads", [4, 8])
@parametrize("embed_dim, num_attn_heads", [(96, 16), (128, 32)])
@parametrize("batch", [1, 2])
@torch.no_grad()
def test_qwen2_decoder_layer(batch, embed_dim, num_attn_heads, num_kv_heads):
    seq_len, intermediate_size, base, eps = 123, 256, 23432, 1e-6
    head_dim = embed_dim // num_attn_heads
    nnm_rope = RoPE(rotary_inv_freq(head_dim, base))
    nnm_decoder = Qwen2DecoderLayer(
        embed_dim=embed_dim,
        intermediate_size=intermediate_size,
        num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads,
        eps=eps,
    )
    config = cfg.Qwen2Config(
        hidden_size=embed_dim,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_attention_heads=num_attn_heads,
        num_key_value_heads=num_kv_heads,
        rms_norm_eps=eps,
        head_dim=head_dim,
        rope_theta=base,
    )
    config._attn_implementation = "sdpa"
    hf_rope = qwen2.Qwen2RotaryEmbedding(config)
    hf_decoder = qwen2.Qwen2DecoderLayer(config, 0)

    init_decoder_layer(nnm_decoder, hf_decoder)

    x = torch.randn(batch, seq_len, embed_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)
    attn_mask = None

    nnm_cos, nnm_sin = nnm_rope(torch.arange(seq_len))
    nnm_o = nnm_decoder(x, rotary_embeddings=(nnm_cos, nnm_sin), attn_mask=attn_mask)
    hf_o = hf_decoder(x, position_embeddings=(hf_cos, hf_sin), attention_mask=attn_mask)[0]
    if nnm_o.shape != hf_o.shape:
        nnm_o = nnm_o[0]
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


def init_qwen2_backbone(nnm_backbone, hf_backbone):
    hf_backbone.embed_tokens.weight = nnm_backbone.token_embeds.weight
    assert len(nnm_backbone.layers) == len(hf_backbone.layers)
    for nnm_decoder_layer, hf_decoder_layer in zip(nnm_backbone.layers, hf_backbone.layers):
        init_decoder_layer(nnm_decoder_layer, hf_decoder_layer)
    hf_backbone.norm.weight = nnm_backbone.norm.weight


def random_causal_mask(x):
    batch, seq_len = x.shape
    attn_mask = torch.ones_like(x, dtype=torch.float32)
    min_idx = max(1, (seq_len - 1) // 2)
    for b in range(batch):
        idx = random.randint(min_idx, seq_len - 1)
        attn_mask[b, idx:] = 0
    return attn_mask


@parametrize("num_kv_heads", [4, 8])
@parametrize("embed_dim, num_attn_heads", [(96, 16), (128, 32)])
@parametrize("batch", [1, 2])
@torch.no_grad()
def test_qwen2_backbone(
    batch,
    embed_dim,
    num_attn_heads,
    num_kv_heads,
):
    max_seq_len, seq_len, intermediate_size, base, eps = 512, 123, 256, 23432, 1e-6
    vocab_size, num_hidden_layers, sliding_window = 1234, 6, 1024
    head_dim = embed_dim // num_attn_heads
    nnm_backbone = Qwen2Model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        padding_idx=0,
        rms_norm_eps=eps,
        num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads,
        rope_base=base,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        sliding_window=sliding_window,
    )
    config = cfg.Qwen2Config(
        hidden_size=embed_dim,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_attention_heads=num_attn_heads,
        num_key_value_heads=num_kv_heads,
        sliding_window=sliding_window,
        rms_norm_eps=eps,
        head_dim=head_dim,
        rope_theta=base,
        max_position_embeddings=max_seq_len,
        vocab_size=vocab_size,
        use_cache=False,
        num_hidden_layers=num_hidden_layers,
    )
    config._attn_implementation = "sdpa"
    hf_backbone = qwen2.Qwen2Model(config)

    init_qwen2_backbone(nnm_backbone, hf_backbone)

    x = torch.randint(0, vocab_size, (batch, seq_len))
    attn_mask = random_causal_mask(x)

    nnm_o = nnm_backbone(x, attn_mask=attn_mask)
    hf_o = hf_backbone(input_ids=x, attention_mask=attn_mask)[0]
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_qwen2_incremental_decoding_with_external_cache():
    model = Qwen2Model(
        vocab_size=100,
        embed_dim=32,
        max_seq_len=64,
        padding_idx=0,
        rms_norm_eps=1e-6,
        num_attn_heads=4,
        num_kv_heads=2,
        rope_base=10000.0,
        intermediate_size=64,
        num_hidden_layers=3,
        sliding_window=None,
    ).eval()
    cache = Cache([KVCache(64) for _ in range(model.num_hidden_layers)])
    input_ids = torch.randint(0, 100, (2, 9))

    full = model(input_ids)
    chunks = [model(input_ids[:, :4], cache=cache), model(input_ids[:, 4:7], cache=cache)]
    chunks.append(model(input_ids[:, 7:], cache=cache))

    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-5, rtol=1e-5)
