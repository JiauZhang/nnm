import torch
import torch.nn as nn
import torch.nn.functional as F

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.cache import KVCache
from nnm.layers.attention_mask import make_causal_mask
from nnm.models.pretrained import PretrainedModel


class GPT2Attention(nn.Module):
    def __init__(self, *, embed_dim, num_heads, use_cache=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_cache = use_cache
        self.attn_scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, attn_mask=None, cache=None):
        batch, seq_len, _ = x.shape

        query, key, value = (
            part.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            for part in self.qkv_proj(x).chunk(3, dim=-1)
        )

        if self.use_cache and cache is not None:
            cache_len = cache.kv_len
            key, value = cache.update(key, value)
            if attn_mask is None and seq_len > 1:
                attn_mask = make_causal_mask(seq_len, key.shape[-2], cache_len, query.device, query.dtype)
            is_causal = False
        else:
            is_causal = attn_mask is None

        o = scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal, scale=self.attn_scale
        )
        return self.out_proj(o.transpose(1, 2).reshape(batch, seq_len, self.embed_dim))


class GPT2MLP(nn.Module):
    def __init__(self, *, embed_dim, intermediate_size):
        super().__init__()
        self.fc_in = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.fc_out = nn.Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, x):
        return self.fc_out(F.gelu(self.fc_in(x), approximate='tanh'))


class GPT2DecoderLayer(nn.Module):
    def __init__(self, *, embed_dim, intermediate_size, num_heads, eps, use_cache=False):
        super().__init__()
        self.attn = GPT2Attention(embed_dim=embed_dim, num_heads=num_heads, use_cache=use_cache)
        self.mlp = GPT2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size)
        self.norm_1 = nn.LayerNorm(embed_dim, eps=eps)
        self.norm_2 = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x, attn_mask=None, cache=None):
        x = x + self.attn(self.norm_1(x), attn_mask=attn_mask, cache=cache)
        return x + self.mlp(self.norm_2(x))


class GPT2Backbone(nn.Module):
    def __init__(self, *, vocab_size, embed_dim, intermediate_size, max_seq_len, num_layers, num_heads, eps, use_cache=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_cache = use_cache

        self.token_embeds = nn.Embedding(vocab_size, embed_dim)
        self.position_embeds = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                GPT2DecoderLayer(
                    embed_dim=embed_dim,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    eps=eps,
                    use_cache=use_cache,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.caches = [KVCache() for _ in range(num_layers)] if use_cache else None

    def forward(self, input_ids, attn_mask=None):
        batch, seq_len = input_ids.shape
        cache_len = self.caches[0].kv_len if self.use_cache and self.caches else 0

        if attn_mask is not None:
            attn_mask = make_causal_mask(
                seq_len, seq_len + cache_len, cache_len, attn_mask.device, attn_mask.dtype, attn_mask=attn_mask
            )

        positions = torch.arange(cache_len, cache_len + seq_len, device=input_ids.device)
        h = self.token_embeds(input_ids) + self.position_embeds(positions)

        for decoder_layer, cache in zip(self.layers, self.caches or [None] * self.num_layers):
            h = decoder_layer(h, attn_mask=attn_mask, cache=cache)

        return self.norm(h)


class GPT2LM(PretrainedModel):
    def __init__(self, config, use_cache=False):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', True)
        embed_dim = config.n_embd
        intermediate_size = getattr(config, 'n_inner', None) or 4 * embed_dim

        self.backbone = GPT2Backbone(
            vocab_size=config.vocab_size,
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            max_seq_len=config.n_ctx,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            eps=config.layer_norm_epsilon,
            use_cache=use_cache,
        )
        self.lm_head = None if self.tie_word_embeddings else nn.Linear(embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids, attn_mask=None):
        h = self.backbone(input_ids, attn_mask=attn_mask)
        if self.lm_head is not None:
            return self.lm_head(h)
        return torch.matmul(h, self.backbone.token_embeds.weight.t())

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()
