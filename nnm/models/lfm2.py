import torch
from functools import partial
from torch import nn
from nnm.layers.attention import RotarySelfAttention
from nnm.layers.mlp import GatedMLP
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.cache import KVCache, StateCache
from nnm.models.pretrained import PretrainedModel
from nnm.layers.conv import ShortConv


class Lfm2MLP(GatedMLP):
    def __init__(
        self,
        *,
        hidden_size,
        intermediate_size,
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=256,
    ):
        if block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if block_ffn_dim_multiplier is not None:
                intermediate_size = int(block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = block_multiple_of * ((intermediate_size + block_multiple_of - 1) // block_multiple_of)
        super().__init__(hidden_size, intermediate_size)


class Lfm2Attention(RotarySelfAttention):
    def __init__(self, *, config):
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        norm_eps = getattr(config, "norm_eps", 1e-6)
        super().__init__(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            qk_norm=True,
            norm_layer=partial(nn.RMSNorm, eps=norm_eps, elementwise_affine=True),
        )
        self.use_cache = getattr(config, "use_cache", False)

    def forward(self, hidden_states, rotary_embeddings, attn_mask=None, cache=None):
        return super().forward(
            hidden_states,
            rotary_embeddings,
            cache=cache if self.use_cache else None,
            attn_mask=attn_mask,
        )


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, *, config, layer_idx, is_attention_layer, head_dim=None):
        super().__init__()
        self.is_attention_layer = is_attention_layer
        hidden_size = config.hidden_size
        norm_eps = getattr(config, "norm_eps", 1e-6)

        if is_attention_layer:
            self.self_attn = Lfm2Attention(config=config)
        else:
            self.conv = ShortConv(
                hidden_size,
                getattr(config, "conv_L_cache", 3),
                getattr(config, "conv_bias", False),
                getattr(config, "use_cache", False),
            )
        intermediate_size = getattr(config, "intermediate_size", None) or getattr(
            config, "block_ff_dim", config.hidden_size * 4
        )
        self.feed_forward = Lfm2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_auto_adjust_ff_dim=getattr(config, "block_auto_adjust_ff_dim", True),
            block_ffn_dim_multiplier=getattr(config, "block_ffn_dim_multiplier", 1.0),
            block_multiple_of=getattr(config, "block_multiple_of", 256),
        )
        self.operator_norm = nn.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)

    def forward(self, hidden_states, rotary_embeddings=None, attn_mask=None, cache=None):
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                rotary_embeddings=rotary_embeddings,
                attn_mask=attn_mask,
                cache=cache,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                cache=cache,
                attn_mask=attn_mask,
            )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

        return hidden_states


class Lfm2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.use_cache = getattr(config, "use_cache", False)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.position_encoder = RoPE(rotary_inv_freq(self.head_dim, rope_theta))

        full_attn_idxs = getattr(config, "full_attn_idxs", list(range(config.num_hidden_layers)))
        self.layer_types = ["full_attention" if i in full_attn_idxs else "conv" for i in range(config.num_hidden_layers)]

        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    is_attention_layer=self.layer_types[layer_idx] == "full_attention",
                    head_dim=self.head_dim,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=getattr(config, "norm_eps", 1e-5), elementwise_affine=True)
        if self.use_cache:
            self.caches = []
            for layer_type in self.layer_types:
                if layer_type == "full_attention":
                    self.caches.append(KVCache())
                else:
                    self.caches.append(StateCache(state_size=getattr(config, "conv_L_cache", 3)))
        else:
            self.caches = None

    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        cache_len = next(
            (cache.kv_len for cache in self.caches or [] if isinstance(cache, KVCache)),
            0,
        )
        positions = torch.arange(cache_len, cache_len + hidden_states.shape[1], device=hidden_states.device)
        rotary_embeddings = self.position_encoder(positions)

        for decoder_layer, cache in zip(self.layers, self.caches or [None] * len(self.layers)):
            hidden_states = decoder_layer(
                hidden_states,
                rotary_embeddings=rotary_embeddings,
                attn_mask=attn_mask,
                cache=cache,
            )

        hidden_states = self.embedding_norm(hidden_states)
        return hidden_states


class Lfm2LM(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.backbone = Lfm2Model(config)
        if self.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attn_mask=None):
        output_embeds = self.backbone(input_ids, attn_mask=attn_mask)
        if self.lm_head is not None:
            logits = self.lm_head(output_embeds)
        else:
            logits = torch.matmul(output_embeds, self.backbone.embed_tokens.weight.t())
        return logits

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()
