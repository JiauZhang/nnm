import torch
from torch import nn
from nnm.layers.mlp import GateMLP
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.transformer import SelfAttention, TransformerBlock
from nnm.cache import KVCache
from nnm.models.pretrained import PretrainedModel


class Qwen2MLP(GateMLP):
    def __init__(self, *, embed_dim, intermediate_size):
        super().__init__(embed_dim, intermediate_size)


class Qwen2DecoderLayer(TransformerBlock):
    def __init__(
        self,
        *,
        position_encoder,
        embed_dim,
        intermediate_size,
        num_attn_heads,
        num_kv_heads,
        eps,
        use_cache=False,
    ):
        attention = SelfAttention(
            embed_dim,
            num_attn_heads,
            num_kv_heads,
            qkv_bias=True,
            o_bias=False,
            rope=position_encoder,
        )
        super().__init__(
            embed_dim,
            eps,
            attention,
            Qwen2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size),
            norm_factory=lambda dim: nn.RMSNorm(dim, eps=eps, elementwise_affine=True),
        )


class Qwen2Backbone(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        embed_dim,
        max_seq_len,
        padding_idx,
        num_hidden_layers,
        rms_norm_eps,
        rope_base,
        num_attn_heads,
        num_kv_heads,
        intermediate_size,
        sliding_window,
        use_cache=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_attn_heads
        self.sliding_window = sliding_window
        self.padding_idx = padding_idx
        self.num_hidden_layers = num_hidden_layers
        self.use_cache = use_cache
        self.token_embeds = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_encoder = RoPE(rotary_inv_freq(self.head_dim, rope_base))
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    position_encoder=self.position_encoder,
                    embed_dim=embed_dim,
                    intermediate_size=intermediate_size,
                    num_attn_heads=num_attn_heads,
                    num_kv_heads=num_kv_heads,
                    eps=rms_norm_eps,
                    use_cache=use_cache,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(embed_dim, eps=rms_norm_eps, elementwise_affine=True)
        self.caches = [KVCache() for _ in range(num_hidden_layers)] if use_cache else None

    def forward(self, input_ids, attn_mask=None):
        output_embeds = self.token_embeds(input_ids)
        cache_len = self.caches[0].kv_len if self.use_cache and self.caches else 0
        if attn_mask is not None:
            batch, seq_len = attn_mask.shape
            attn_mask = make_causal_mask(
                seq_len,
                seq_len + cache_len,
                cache_len,
                attn_mask.device,
                attn_mask.dtype,
                attn_mask=attn_mask,
                sliding_window=self.sliding_window,
            )
        for decoder_layer, cache in zip(self.layers, self.caches or [None] * len(self.layers)):
            output_embeds = decoder_layer(output_embeds, attn_mask=attn_mask, cache=cache)
        output_embeds = self.norm(output_embeds)
        return output_embeds


class Qwen2LM(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.backbone = Qwen2Backbone(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            max_seq_len=config.max_position_embeddings,
            padding_idx=getattr(config, "pad_token_id", 0),
            num_hidden_layers=config.num_hidden_layers,
            num_attn_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_base=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            intermediate_size=config.intermediate_size,
            sliding_window=config.sliding_window,
            use_cache=getattr(config, "use_cache", False),
        )
        if self.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attn_mask=None):
        output_embeds = self.backbone(input_ids, attn_mask=attn_mask)
        if self.lm_head is not None:
            logits = self.lm_head(output_embeds)
        else:
            logits = torch.matmul(output_embeds, self.backbone.token_embeds.weight.t())
        return logits

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()
