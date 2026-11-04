import torch
from torch import nn

from nnm.cache import KVCache, StateCache
from nnm.layers.conv import ShortConv
from nnm.layers.mlp import SwiGLU
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.transformer import SelfAttention
from nnm.models.pretrained import PretrainedModel


class Attention(SelfAttention):
    def __init__(self, config, layer_idx):
        super().__init__(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            q_norm=True,
            k_norm=True,
            norm_eps=config.norm_eps,
            output_gate="head",
            sliding_window=config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None,
        )
        self.use_cache = getattr(config, "use_cache", False)

    def forward(self, x, cos, sin, cache=None, attn_mask=None):
        cache = cache if self.use_cache else None
        return super().forward(x, cos, sin, cache, attn_mask)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] != "conv"
        self.sandwich = config.use_sandwich_norm
        if self.is_attention:
            self.self_attn = Attention(config, layer_idx)
        else:
            self.conv = ShortConv(
                config.hidden_size, config.conv_L_cache, config.conv_bias, getattr(config, "use_cache", False)
            )
        self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size)
        self.operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if self.sandwich:
            self.post_operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x, cos, sin, cache=None, attn_mask=None):
        residual = x
        hidden = self.operator_norm(x)
        hidden = (
            self.self_attn(hidden, cos, sin, cache, attn_mask) if self.is_attention else self.conv(hidden, cache, attn_mask)
        )
        if self.sandwich:
            hidden = self.post_operator_norm(hidden)
        hidden = hidden + residual

        feed_forward = self.feed_forward(self.ffn_norm(hidden))
        if self.sandwich:
            feed_forward = self.post_ffn_norm(feed_forward)
        return hidden + feed_forward


class Lfm3Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_cache = getattr(config, "use_cache", False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.rotary = nn.ModuleDict(
            {
                layer_type: RoPE(
                    rotary_inv_freq(
                        config.head_dim,
                        config.rope_parameters[layer_type]["rope_theta"],
                        int(config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0) * config.head_dim),
                    )
                )
                for layer_type in config.rope_parameters
            }
        )
        self.layers = nn.ModuleList([DecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if self.use_cache:
            self.caches = [
                KVCache() if layer_type != "conv" else StateCache(config.conv_L_cache) for layer_type in config.layer_types
            ]
        else:
            self.caches = None
        self.position_offset = 0

    def forward(self, input_ids=None, inputs_embeds=None, attn_mask=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        seq_len = hidden.shape[1]
        offset = self.position_offset
        positions = torch.arange(offset, offset + seq_len, device=hidden.device)

        for index, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[index]
            cos, sin = self.rotary[layer_type](positions) if layer_type != "conv" else (None, None)
            hidden = layer(hidden, cos, sin, self.caches[index] if self.use_cache else None, attn_mask)

        if self.use_cache:
            self.position_offset += seq_len
        return self.embedding_norm(hidden)


class Lfm3EmbeddingBackbone(Lfm3Backbone):
    pass


class Lfm3LM(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.backbone = Lfm3Backbone(config)
        self.lm_head = None if self.tie_word_embeddings else nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attn_mask=None):
        hidden = self.backbone(input_ids, attn_mask=attn_mask)
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return torch.matmul(hidden, self.backbone.embed_tokens.weight.t())

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()
        self.backbone.position_offset = 0
