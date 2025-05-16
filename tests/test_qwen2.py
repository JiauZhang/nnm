import torch, pytest
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from transformers.models.qwen2 import configuration_qwen2 as cfg
from nnm.layers.rope import QwenRoPE
from nnm.models.qwen2 import Qwen2Attention, Qwen2MLP, Qwen2DecoderLayer, Qwen2Backbone
from nnm.layers.norm import Qwen2RMSNorm

def init_linear_weight(src, dst, bias=True):
    assert dst.weight.shape == src.weight.shape
    dst.weight = src.weight
    if bias:
        assert dst.bias.shape == src.bias.shape
        dst.bias = src.bias

def init_kv_linear_weight(layer, w, b):
    assert w.shape == layer.weight.shape
    assert b.shape == layer.bias.shape
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)

@pytest.mark.parametrize(
    'batch, seq_len, max_seq_len, embed_dim, num_attn_heads, num_kv_heads, base', [
        (1, 234, 512, 32, 8, 4, 2333), (2, 345, 768, 64, 8, 8, 3456),
    ]
)
@torch.no_grad()
def test_qwen2_attn(batch, seq_len, max_seq_len, embed_dim, num_attn_heads, num_kv_heads, base):
    x = torch.randn(batch, seq_len, embed_dim)
    position_ids = torch.arange(seq_len)
    head_dim = embed_dim // num_attn_heads

    rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=head_dim, base=base)
    nnm_cos = rope.cos[position_ids, :]
    nnm_sin = rope.sin[position_ids, :]
    nnm_attn = Qwen2Attention(
        embed_dim=embed_dim, num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads, position_encoder=rope,
    )

    config = cfg.Qwen2Config(
        hidden_size=embed_dim, num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads,
        head_dim=head_dim, rope_theta=base, max_position_embeddings=max_seq_len,
    )
    hf_attn = qwen2.Qwen2Attention(config, 0)

    init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj)
    nnm_q = nnm_attn.q_proj(x)
    hf_q = hf_attn.q_proj(x)
    assert nnm_q.shape == hf_q.shape
    assert (nnm_q - hf_q).abs().mean() < 1e-5

    init_linear_weight(nnm_attn.o_proj, hf_attn.o_proj, bias=False)
    nnm_o = nnm_attn.o_proj(x)
    hf_o = hf_attn.o_proj(x)
    assert nnm_o.shape == hf_o.shape
    assert (nnm_o - hf_o).abs().mean() < 1e-5

    kv_embed_dim = head_dim * num_kv_heads
    init_kv_linear_weight(hf_attn.k_proj, nnm_attn.kv_proj.weight[:kv_embed_dim, :], nnm_attn.kv_proj.bias[:kv_embed_dim])
    init_kv_linear_weight(hf_attn.v_proj, nnm_attn.kv_proj.weight[kv_embed_dim:, :], nnm_attn.kv_proj.bias[kv_embed_dim:])
    nnm_kv = nnm_attn.kv_proj(x)
    nnm_k, nnm_v = nnm_kv.split(kv_embed_dim, dim=-1)
    hf_k, hf_v = hf_attn.k_proj(x), hf_attn.v_proj(x)
    assert nnm_k.shape == hf_k.shape and nnm_v.shape == hf_v.shape
    assert (nnm_k - hf_k).abs().mean() < 1e-5
    assert (nnm_v - hf_v).abs().mean() < 1e-5

    hf_rope = qwen2.Qwen2RotaryEmbedding(config=config)
    position_ids = position_ids.unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)

    # recover nnm sin embeds trick to normal format
    nnm_sin[:, :(head_dim//2)] = -nnm_sin[:, :(head_dim//2)]
    nnm_sin = nnm_sin.unsqueeze(0)
    nnm_cos = nnm_cos.unsqueeze(0)

    assert nnm_cos.shape == hf_cos.shape and nnm_sin.shape == hf_sin.shape
    assert (nnm_cos - hf_cos).abs().mean() < 1e-5
    assert (nnm_sin - hf_sin).abs().mean() < 1e-5

    nnm_o = nnm_attn(x)
    # hf Qwen2Attention has attention dropout
    hf_attn.eval()

    hf_o, hf_attn_weight = hf_attn(x, (hf_cos, hf_sin), None)
    assert nnm_o.shape == hf_o.shape
    assert torch.abs(nnm_o - hf_o).mean() < 1e-5

@pytest.mark.parametrize('batch, seq_len, embed_dim, eps', [(1, 123, 64, 1e-6), (2, 233, 128, 1e-8)])
@torch.no_grad()
def test_qwen2_rms_norm(batch, seq_len, embed_dim, eps):
    hf_norm = qwen2.Qwen2RMSNorm(embed_dim, eps=eps)
    nnm_norm = Qwen2RMSNorm(embed_dim, eps=eps)
    assert nnm_norm.weight.shape == hf_norm.weight.shape

    nnm_norm.weight = torch.nn.Parameter(torch.randn(*nnm_norm.weight.shape))
    x = torch.randn(batch, seq_len, embed_dim)
    hf_norm.weight = nnm_norm.weight
    nnm_o = nnm_norm(x)
    hf_o = hf_norm(x)
    assert nnm_o.shape == hf_o.shape
    assert (nnm_o - hf_o).abs().mean() < 1e-5

def init_mlp_weight(src, dst):
    init_linear_weight(src.gate_proj, dst.gate_proj, bias=False)
    init_linear_weight(src.up_proj, dst.up_proj, bias=False)
    init_linear_weight(src.down_proj, dst.down_proj, bias=False)

@pytest.mark.parametrize('batch, seq_len, embed_dim, intermediate_size', [(1, 123, 64, 256), (2, 233, 128, 384)])
@torch.no_grad()
def test_qwen2_mlp(batch, seq_len, embed_dim, intermediate_size):
    config = cfg.Qwen2Config(hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu")
    hf_mlp = qwen2.Qwen2MLP(config)
    nnm_mlp = Qwen2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size)

    init_mlp_weight(nnm_mlp, hf_mlp)
    x = torch.randn(batch, seq_len, embed_dim)
    nnm_o = nnm_mlp(x)
    hf_o = hf_mlp(x)
    assert (nnm_o - hf_o).abs().mean() < 1e-5

def init_decoder_layer(nnm_decoder, hf_decoder, kv_embed_dim):
    init_mlp_weight(nnm_decoder.mlp, hf_decoder.mlp)
    init_linear_weight(nnm_decoder.norm_1, hf_decoder.input_layernorm, bias=False)
    init_linear_weight(nnm_decoder.norm_2, hf_decoder.post_attention_layernorm, bias=False)
    nnm_attn, hf_attn = nnm_decoder.attn, hf_decoder.self_attn
    init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj)
    init_linear_weight(nnm_attn.o_proj, hf_attn.o_proj, bias=False)
    init_kv_linear_weight(hf_attn.k_proj, nnm_attn.kv_proj.weight[:kv_embed_dim, :], nnm_attn.kv_proj.bias[:kv_embed_dim])
    init_kv_linear_weight(hf_attn.v_proj, nnm_attn.kv_proj.weight[kv_embed_dim:, :], nnm_attn.kv_proj.bias[kv_embed_dim:])
    # hf Qwen2Attention has attention dropout
    hf_decoder.eval()

@pytest.mark.parametrize(
    'batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps', [
        (1, 123, 512, 96, 256, 16, 4, 23432, 1e-6), (2, 233, 768, 128, 384, 32, 8, 10000, 1e-7),
    ]
)
@torch.no_grad()
def test_qwen2_decoder_layer(batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps):
    head_dim = embed_dim // num_attn_heads
    kv_embed_dim = head_dim * num_kv_heads
    nnm_rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=head_dim, base=base)
    nnm_decoder = Qwen2DecoderLayer(
        position_encoder=nnm_rope, embed_dim=embed_dim, intermediate_size=intermediate_size,
        num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads, eps=eps,
    )
    config = cfg.Qwen2Config(
        hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu",
        num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads,
        rms_norm_eps=eps, head_dim=head_dim, rope_theta=base,
    )
    hf_rope = qwen2.Qwen2RotaryEmbedding(config)
    hf_decoder = qwen2.Qwen2DecoderLayer(config, 0)

    init_decoder_layer(nnm_decoder, hf_decoder, kv_embed_dim)

    x = torch.randn(batch, seq_len, embed_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)
    attn_mask = None

    nnm_o = nnm_decoder(x, attn_mask=attn_mask)
    hf_o = hf_decoder(x, position_embeddings=(hf_cos, hf_sin), attention_mask=attn_mask)[0]
    assert nnm_o.shape == hf_o.shape
    assert torch.abs(nnm_o - hf_o).mean() < 1e-5

@pytest.mark.parametrize(
    ','.join([
        'batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps',
        'vocab_size, num_hidden_layers',
    ]), [
        (1, 123, 512, 96, 256, 16, 4, 23432, 1e-6, 1234, 6),
        (2, 233, 768, 128, 384, 32, 8, 10000, 1e-7, 2345, 9),
    ]
)
@torch.no_grad()
def test_qwen2_backbone(
    batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps,
    vocab_size, num_hidden_layers,
):
    head_dim = embed_dim // num_attn_heads
    kv_embed_dim = head_dim * num_kv_heads
    nnm_backbone = Qwen2Backbone(
        vocab_size=vocab_size, embed_dim=embed_dim, max_seq_len=max_seq_len, padding_idx=0, rms_norm_eps=eps,
        num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads, rope_base=base, intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
    )
    config = cfg.Qwen2Config(
        hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu",
        num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads,
        rms_norm_eps=eps, head_dim=head_dim, rope_theta=base, max_position_embeddings=max_seq_len,
        vocab_size=vocab_size, use_cache=False, num_hidden_layers=num_hidden_layers,
    )
    hf_backbone = qwen2.Qwen2Model(config)

    hf_backbone.embed_tokens.weight = nnm_backbone.token_embeds.weight
    assert len(nnm_backbone.layers) == len(hf_backbone.layers)
    for nnm_decoder_layer, hf_decoder_layer in zip(nnm_backbone.layers, hf_backbone.layers):
        init_decoder_layer(nnm_decoder_layer, hf_decoder_layer, kv_embed_dim)
    hf_backbone.norm.weight = nnm_backbone.norm.weight

    x = torch.randint(0, vocab_size, (batch, seq_len))
    attn_mask = torch.zeros((1, 1, 1, 1))

    nnm_o = nnm_backbone(x, attn_mask)
    hf_o = hf_backbone(input_ids=x, attention_mask=attn_mask)[0]
    assert nnm_o.shape == hf_o.shape
    assert torch.abs(nnm_o - hf_o).mean() < 1e-5