import torch
from functools import partial
from torch import nn

from nnm.cache import KVCache
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.models.qwen3 import make_qwen3_layer


class Qwen3ASRTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        norm_layer = partial(RMSNorm, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([make_qwen3_layer(config, norm_layer=norm_layer) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = RoPE(rotary_inv_freq(config.head_dim, config.rope_theta))
        self.caches = None

    def prepare_cache(self):
        self.caches = [KVCache() for _ in self.layers]

    def clear_cache(self):
        if self.caches:
            for cache in self.caches:
                cache.clear()

    def forward(self, inputs_embeds, attn_mask=None, use_cache=False):
        if use_cache and self.caches is None:
            self.prepare_cache()
        offset = self.caches[0].kv_len if use_cache and not self.caches[0].is_empty() else 0
        positions = torch.arange(offset, offset + inputs_embeds.shape[1], device=inputs_embeds.device)
        cos, sin = self.rotary(positions)

        hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                rotary_embeddings=(cos, sin),
                cache=self.caches[index] if use_cache else None,
                attn_mask=attn_mask,
            )
        return self.norm(hidden)
