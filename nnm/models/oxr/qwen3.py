import torch
from functools import partial
from torch import nn

from nnm.cache import KVCache
from nnm.layers.attention import RotarySelfAttention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.mlp import GatedMLP
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.transformer import TransformerBlock


class Qwen3Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.use_cache = getattr(config, "use_cache", False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.position_encoder = RoPE(rotary_inv_freq(head_dim, config.rope_theta))
        self.layers = nn.ModuleList([self._make_layer(config, head_dim) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.caches = [KVCache() for _ in range(config.num_hidden_layers)] if self.use_cache else None

    def _make_layer(self, config, head_dim):
        norm_layer = partial(nn.RMSNorm, eps=config.rms_norm_eps, elementwise_affine=True)
        attention = RotarySelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            qk_norm=True,
            norm_layer=norm_layer,
        )
        return TransformerBlock(
            config.hidden_size,
            attention,
            GatedMLP(config.hidden_size, config.intermediate_size),
            norm_layer=norm_layer,
        )

    def forward(self, inputs_embeds, attn_mask=None):
        cache_len = self.caches[0].kv_len if self.use_cache and self.caches else 0
        if attn_mask is not None:
            batch, seq_len = attn_mask.shape
            attn_mask = make_causal_mask(
                seq_len, seq_len + cache_len, cache_len, attn_mask.device, attn_mask.dtype, attn_mask=attn_mask
            )
        positions = torch.arange(cache_len, cache_len + inputs_embeds.shape[1], device=inputs_embeds.device)
        rotary_embeddings = self.position_encoder(positions)
        for layer, cache in zip(self.layers, self.caches or [None] * len(self.layers)):
            inputs_embeds = layer(
                inputs_embeds,
                rotary_embeddings=rotary_embeddings,
                attn_mask=attn_mask,
                cache=cache,
            )
        return self.norm(inputs_embeds)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = config.tie_word_embeddings
        self.backbone = Qwen3Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(self, inputs_embeds, attn_mask=None):
        hidden_states = self.backbone(inputs_embeds, attn_mask)
        return self.lm_head(hidden_states)

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()
