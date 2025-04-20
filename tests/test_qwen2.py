import torch, pytest
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from transformers.models.qwen2 import configuration_qwen2 as cfg
from nnm.layers.positional import QwenRoPE
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
        'batch, seq_len, embed_dim, num_attn_heads, num_kv_heads', [(1, 16, 32, 8, 4), (2, 32, 64, 32, 8)])
def test_qwen2_attn(batch, seq_len, embed_dim, num_attn_heads, num_kv_heads):
    rope = QwenRoPE(max_seq_len=256, embed_dim=embed_dim//num_attn_heads)
    seq_idx = rope.seq_idx(False, 0, seq_len)
    sin_pe = rope.sin[seq_idx, :].unsqueeze(0)
    cos_pe = rope.cos[seq_idx, :].unsqueeze(0)
    config = cfg.Qwen2Config(
        hidden_size=embed_dim, num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads)
    hf_qwen2_attn = qwen2.Qwen2Attention(config, 0)
    attn = Qwen2Attention(
        embed_dim=embed_dim, num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads, position_encoder=rope,
    )

    __init_linear_weight(attn.q_proj, hf_qwen2_attn.q_proj)
    __init_linear_weight(attn.o_proj, hf_qwen2_attn.o_proj)
    head_dim = embed_dim // num_attn_heads
    kv_embed_dim = head_dim * num_kv_heads
    __init_kv_linear_weight(hf_qwen2_attn.k_proj, attn.kv_proj.weight[:kv_embed_dim, :], attn.kv_proj.bias[:kv_embed_dim])
    __init_kv_linear_weight(hf_qwen2_attn.v_proj, attn.kv_proj.weight[kv_embed_dim:, :], attn.kv_proj.bias[kv_embed_dim:])
    hf_qwen2_attn.eval()

    x = torch.randn(batch, seq_len, embed_dim)
    hf_o, hf_attn_weight = hf_qwen2_attn(x, (cos_pe, sin_pe), None)
    o = attn(x)

    assert o.shape == hf_o.shape
    assert torch.abs(o - hf_o).mean() < 1e-5