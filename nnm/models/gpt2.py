import torch
from functools import partial
from torch import nn

from nnm.cache import KVCache
from nnm.layers.attention import SelfAttention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.mlp import MLP
from nnm.layers.transformer import TransformerBlock
from nnm.models.pretrained import PretrainedModel


class GPT2Attention(SelfAttention):
    def __init__(self, *, embed_dim, num_heads, max_seq_len, use_cache=False):
        super().__init__(embed_dim, num_heads, qkv_bias=True, o_bias=True)
        self.use_cache = use_cache
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, attn_mask=None, cache=None):
        return super().forward(x, attn_mask=attn_mask, cache=cache if self.use_cache else None)


class GPT2Model(nn.Module):
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
                TransformerBlock(
                    embed_dim,
                    GPT2Attention(embed_dim=embed_dim, num_heads=num_heads, max_seq_len=max_seq_len, use_cache=use_cache),
                    MLP(embed_dim, intermediate_size, activation="gelu_pytorch_tanh", bias=True),
                    norm_layer=partial(nn.LayerNorm, eps=eps),
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
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        embed_dim = config.n_embd
        intermediate_size = getattr(config, "n_inner", None) or 4 * embed_dim

        self.backbone = GPT2Model(
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
