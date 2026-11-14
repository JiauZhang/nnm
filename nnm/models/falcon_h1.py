import torch
from torch import nn

from nnm.layers.attention import RotarySelfAttention
from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.mamba import Mamba2Mixer
from nnm.layers.mlp import GatedMLP
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, rotary_inv_freq


class FalconH1DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = Mamba2Mixer(config)
        head_dim = config.head_dim
        self.self_attn = RotarySelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
        )
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = GatedMLP(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)

    def forward(self, hidden, rotary_embeddings=None, attn_mask=None, conv_state=None, ssm_state=None, cache=None):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        mamba_out, conv_state, ssm_state = self.mamba(hidden, conv_state, ssm_state)
        attn_out = self.self_attn(hidden, rotary_embeddings=rotary_embeddings, attn_mask=attn_mask, cache=cache)
        hidden = residual + mamba_out + attn_out
        residual = hidden
        hidden = self.pre_ff_layernorm(hidden)
        hidden = self.feed_forward(hidden)
        return residual + hidden, conv_state, ssm_state


class FalconH1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        head_dim = config.head_dim
        self.position_encoder = RoPE(rotary_inv_freq(head_dim, config.rope_theta))
        self.layers = nn.ModuleList([FalconH1DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_multiplier = config.embedding_multiplier

    def forward(self, inputs_embeds, attn_mask=None, conv_states=None, ssm_states=None, caches=None):
        inputs_embeds = inputs_embeds * self.embedding_multiplier
        cache_len = caches[0].stored_length if caches is not None else 0
        if attn_mask is not None:
            batch, seq_len = attn_mask.shape
            attn_mask = make_causal_mask(
                seq_len, seq_len + cache_len, cache_len, attn_mask.device, attn_mask.dtype, attn_mask=attn_mask
            )
        positions = torch.arange(cache_len, cache_len + inputs_embeds.shape[1], device=inputs_embeds.device)
        rotary_embeddings = self.position_encoder(positions)
        new_conv_states = []
        new_ssm_states = []
        for i, layer in enumerate(self.layers):
            conv_state = conv_states[i] if conv_states is not None else None
            ssm_state = ssm_states[i] if ssm_states is not None else None
            inputs_embeds, conv_state, ssm_state = layer(
                inputs_embeds,
                rotary_embeddings=rotary_embeddings,
                attn_mask=attn_mask,
                conv_state=conv_state,
                ssm_state=ssm_state,
                cache=caches[i] if caches is not None else None,
            )
            new_conv_states.append(conv_state)
            new_ssm_states.append(ssm_state)
        return self.final_layernorm(inputs_embeds), new_conv_states, new_ssm_states
