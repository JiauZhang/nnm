import torch
from torch import nn
from nnm.layers.rope import QwenRoPE
from nnm.layers.norm import Qwen2RMSNorm

class Qwen2MLP(nn.Module):
    def __init__(self, *, embed_dim, intermediate_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

@torch.no_grad()
def make_causal_attn_mask(attn_mask, kv_cache_len, sliding_window):
    batch, seq_len = attn_mask.shape
    dtype_info = torch.finfo if attn_mask.dtype.is_floating_point else torch.iinfo
    inf_val = dtype_info(attn_mask.dtype).min
    kv_len = seq_len + kv_cache_len
    causal_attn_mask = torch.full((seq_len, kv_len), fill_value=inf_val)
    query_position = torch.arange(kv_cache_len, kv_len)
    key_position = torch.arange(kv_len)
    triu_attn_mask = query_position > key_position.reshape(-1, 1)
    if kv_len > sliding_window:
        sliding_attn_mask = (query_position - sliding_window).reshape(-1, 1) >= key_position.reshape(1, -1)
        triu_attn_mask.bitwise_or_(sliding_attn_mask)
    causal_attn_mask *= triu_attn_mask
    causal_attn_mask = causal_attn_mask[None, :, :].expand(batch, -1, -1)
    padding_mask = causal_attn_mask + attn_mask[:, None, :]
    padding_mask = padding_mask == 0
    causal_attn_mask = causal_attn_mask.masked_fill(padding_mask, inf_val).unsqueeze(1)
    return causal_attn_mask

def update_kv_cache(kv_cache, k_or_v, idx):
    # kv_cache_shape: [batch, num_kv_heads, seq_len, head_dim]
    kv_cache[idx].append(k_or_v)
    k_or_v = torch.cat(kv_cache[idx], dim=-2)
    return k_or_v

class Qwen2Attention(nn.Module):
    def __init__(self, *, embed_dim, num_attn_heads, num_kv_heads, position_encoder, use_kv_cache=False):
        super().__init__()
        assert embed_dim % num_attn_heads == 0
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_attn_heads
        self.num_attn_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_attn_heads // self.num_kv_heads
        self.use_kv_cache = use_kv_cache
        self.position_encoder = position_encoder
        self.attn_scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv_proj = nn.Linear(embed_dim, 2 * num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, kv_cache=None, attn_mask=None):
        batch, seq_len = x.shape[:-1]

        q = self.q_proj(x)
        k, v = self.kv_proj(x).split(self.num_kv_heads * self.head_dim, dim=-1)
        q = q.reshape(batch, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        position = len(kv_cache[0]) if self.use_kv_cache else None
        q = self.position_encoder(q, use_kv_cache=self.use_kv_cache, position=position)
        k = self.position_encoder(k, use_kv_cache=self.use_kv_cache, position=position)

        if self.use_kv_cache:
            k = update_kv_cache(kv_cache, k, 0)
            v = update_kv_cache(kv_cache, v, 1)

        # kv_cache_shape: [batch, num_kv_heads, seq_len, head_dim]
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        attn_weight = q @ k.transpose(2, 3) * self.attn_scale
        if attn_mask is not None: attn_weight += attn_mask.to(dtype=attn_weight.dtype)
        attn_weight = attn_weight.softmax(dim=-1)
        o = attn_weight @ v
        o = self.o_proj(o.transpose(1, 2).reshape(batch, seq_len, self.embed_dim))
        return o

class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self, *, position_encoder, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, eps, use_kv_cache=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.position_encoder = position_encoder
        self.attn = Qwen2Attention(
            embed_dim=embed_dim, num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads,
            position_encoder=position_encoder, use_kv_cache=use_kv_cache,
        )
        self.mlp = Qwen2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size)
        self.norm_1 = Qwen2RMSNorm(embed_dim=embed_dim, eps=eps)
        self.norm_2 = Qwen2RMSNorm(embed_dim=embed_dim, eps=eps)

    def forward(self, x, attn_mask=None, kv_cache=None):
        y = self.norm_1(x)
        y = self.attn(y, attn_mask=attn_mask, kv_cache=kv_cache)
        x = x + y

        y = self.norm_2(x)
        y = self.mlp(y)
        x = x + y

        return x

class Qwen2Backbone(nn.Module):
    def __init__(
        self, *, vocab_size, embed_dim, max_seq_len, padding_idx, num_hidden_layers, rms_norm_eps, rope_base,
        num_attn_heads, num_kv_heads, intermediate_size, sliding_window, use_kv_cache=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_attn_heads
        self.num_kv_groups = num_attn_heads // num_kv_heads
        self.sliding_window = sliding_window
        self.padding_idx = padding_idx
        self.num_hidden_layers = num_hidden_layers
        self.token_embeds = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_encoder = QwenRoPE(max_seq_len=max_seq_len, embed_dim=self.head_dim, base=rope_base)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(
                position_encoder=self.position_encoder, embed_dim=embed_dim, intermediate_size=intermediate_size,
                num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads, eps=rms_norm_eps, use_kv_cache=use_kv_cache,
            ) for _ in range(num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(embed_dim, eps=rms_norm_eps)
        self.kv_cache = [([], []) for _ in range(num_hidden_layers)]

    def forward(self, input_ids, attn_mask=None):
        input_embeds = self.token_embeds(input_ids)
        kv_cache_len = len(self.kv_cache[0][0]) * self.num_kv_groups
        if attn_mask is not None: attn_mask = make_causal_attn_mask(attn_mask, kv_cache_len, self.sliding_window)
        output_embeds = input_embeds
        for decoder_layer, kv_cache in zip(self.layers, self.kv_cache):
            output_embeds = decoder_layer(output_embeds, attn_mask=attn_mask, kv_cache=kv_cache)
        output_embeds = self.norm(output_embeds)
        return output_embeds

class Qwen2LM(nn.Module):
    def __init__(
        self, *, vocab_size, embed_dim, max_seq_len, padding_idx, num_hidden_layers, rms_norm_eps, rope_base,
        num_attn_heads, num_kv_heads, intermediate_size, sliding_window, use_kv_cache=False,
    ):
        super().__init__()
        self.backbone = Qwen2Backbone(
            vocab_size=vocab_size, embed_dim=embed_dim, max_seq_len=max_seq_len, padding_idx=padding_idx,
            num_hidden_layers=num_hidden_layers, num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads,
            rope_base=rope_base, rms_norm_eps=rms_norm_eps, intermediate_size=intermediate_size,
            sliding_window=sliding_window, use_kv_cache=use_kv_cache,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, attn_mask=None):
        output_embeds = self.backbone(input_ids, attn_mask=attn_mask)
        logits = self.lm_head(output_embeds)
        return logits