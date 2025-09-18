def linear(in_features, out_features, bias=True):
    flops = 2 * in_features * out_features
    if bias: flops += out_features
    return flops

def matmul(a_rows, a_cols, b_cols):
    # matmul = mul + add
    return 2 * a_rows * a_cols * b_cols

def embedding(seq_len, embed_dim):
    return seq_len * embed_dim

def attention(seq_len, embed_dim, bias=False):
    qkv_flops = 3 * seq_len * linear(embed_dim, embed_dim, bias=bias)
    attn_flops = matmul(seq_len, embed_dim, seq_len)
    return qkv_flops + attn_flops

def multi_head_attention(seq_len, head_dim, num_heads, bias=False):
    return num_heads * attention(seq_len, head_dim, bias=bias)