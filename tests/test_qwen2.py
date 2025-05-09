import torch, pytest
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from transformers.models.qwen2 import configuration_qwen2 as cfg
from nnm.layers.rope import QwenRoPE
from nnm.models.qwen2 import Qwen2Attention

def __init_linear_weight(src, dst):
    dst.weight = src.weight
    dst.bias = src.bias

def __init_kv_linear_weight(layer, w, b):
    assert w.shape == layer.weight.shape
    assert b.shape == layer.bias.shape
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)

@pytest.mark.parametrize(
    'batch, seq_len, max_seq_len, embed_dim, num_attn_heads, num_kv_heads, base', [
        (1, 234, 512, 32, 8, 4, 2333), (2, 345, 768, 64, 8, 8, 3456),
    ]
)
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

    __init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj)
    nnm_q = nnm_attn.q_proj(x)
    hf_q = hf_attn.q_proj(x)
    assert nnm_q.shape == hf_q.shape
    assert (nnm_q - hf_q).abs().mean() < 1e-5

    __init_linear_weight(nnm_attn.o_proj, hf_attn.o_proj)
    nnm_o = nnm_attn.o_proj(x)
    hf_o = hf_attn.o_proj(x)
    assert nnm_o.shape == hf_o.shape
    assert (nnm_o - hf_o).abs().mean() < 1e-5

    kv_embed_dim = head_dim * num_kv_heads
    __init_kv_linear_weight(hf_attn.k_proj, nnm_attn.kv_proj.weight[:kv_embed_dim, :], nnm_attn.kv_proj.bias[:kv_embed_dim])
    __init_kv_linear_weight(hf_attn.v_proj, nnm_attn.kv_proj.weight[kv_embed_dim:, :], nnm_attn.kv_proj.bias[kv_embed_dim:])
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