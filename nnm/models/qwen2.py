import torch
from functools import partial
from torch import nn
from nnm.layers.attention import RotarySelfAttention
from nnm.layers.mlp import GatedMLP
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.transformer import TransformerBlock
from nnm.models.pretrained import PretrainedModel


class Qwen2MLP(GatedMLP):
    def __init__(self, *, embed_dim, intermediate_size):
        super().__init__(embed_dim, intermediate_size)


class Qwen2DecoderLayer(TransformerBlock):
    def __init__(
        self,
        *,
        embed_dim,
        intermediate_size,
        num_attn_heads,
        num_kv_heads,
        eps,
    ):
        attention = RotarySelfAttention(
            embed_dim,
            num_attn_heads,
            num_kv_heads,
            qkv_bias=True,
            o_bias=False,
        )
        super().__init__(
            embed_dim,
            attention,
            Qwen2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size),
            norm_layer=partial(nn.RMSNorm, eps=eps, elementwise_affine=True),
        )


class Qwen2Model(nn.Module):
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_attn_heads
        self.sliding_window = sliding_window
        self.padding_idx = padding_idx
        self.num_hidden_layers = num_hidden_layers
        self.token_embeds = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_encoder = RoPE(rotary_inv_freq(self.head_dim, rope_base))
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    embed_dim=embed_dim,
                    intermediate_size=intermediate_size,
                    num_attn_heads=num_attn_heads,
                    num_kv_heads=num_kv_heads,
                    eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(embed_dim, eps=rms_norm_eps, elementwise_affine=True)

    def forward(self, input_ids=None, attn_mask=None, inputs_embeds=None, cache=None):
        output_embeds = self.token_embeds(input_ids) if inputs_embeds is None else inputs_embeds
        cache_len = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(cache_len, cache_len + output_embeds.shape[1], device=output_embeds.device)
        rotary_embeddings = self.position_encoder(positions)
        layers = cache.layers if cache is not None else [None] * len(self.layers)
        for decoder_layer, layer_cache in zip(self.layers, layers):
            output_embeds = decoder_layer(
                output_embeds,
                rotary_embeddings=rotary_embeddings,
                attn_mask=attn_mask,
                cache=layer_cache,
            )
        output_embeds = self.norm(output_embeds)
        return output_embeds


class Qwen2LM(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.backbone = Qwen2Model(
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
        )
        if self.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attn_mask=None, cache=None):
        output_embeds = self.backbone(input_ids, attn_mask=attn_mask, cache=cache)
        if self.lm_head is not None:
            logits = self.lm_head(output_embeds)
        else:
            logits = torch.matmul(output_embeds, self.backbone.token_embeds.weight.t())
        return logits
