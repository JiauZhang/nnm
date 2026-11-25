import torch
import torch.nn.functional as F
from torch import nn

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.layers.activation import get_activation
from nnm.pretrained import PretrainedModel


class BartLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, seq_len, past_key_values_length=0, device=None):
        position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=device)
        return super().forward(position_ids + self.offset)


class BartAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = dropout

    def forward(self, hidden_states, key_value_states=None, attention_mask=None, causal=False):
        bsz, tgt_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        current = key_value_states if key_value_states is not None else hidden_states
        src_len = current.shape[1]
        key = self.k_proj(current).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(current).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal and attention_mask is None,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1)
        return self.out_proj(attn_output)


class BartEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1, activation_dropout=0.0, activation_function="gelu"):
        super().__init__()
        self.self_attn = BartAttention(hidden_size, num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.activation_fn = get_activation(activation_function)
        self.dropout = dropout
        self.activation_dropout = activation_dropout

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return self.final_layer_norm(residual + hidden_states)


class BartDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1, activation_dropout=0.0, activation_function="gelu"):
        super().__init__()
        self.self_attn = BartAttention(hidden_size, num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.encoder_attn = BartAttention(hidden_size, num_heads, dropout=dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.activation_fn = get_activation(activation_function)
        self.dropout = dropout
        self.activation_dropout = activation_dropout

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, causal=True)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return self.final_layer_norm(residual + hidden_states)


def _expand_mask(attention_mask, dtype):
    if attention_mask is None:
        return None
    return (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min


class BartEncoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList(
            [
                BartEncoderLayer(
                    config.d_model,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    config.dropout,
                    config.activation_dropout,
                    config.activation_function,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def forward(self, input_ids, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(input_ids.shape[1], device=input_ids.device)
        hidden_states = self.layernorm_embedding(inputs_embeds + embed_pos)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        mask = _expand_mask(attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=mask)
        return hidden_states


class BartDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList(
            [
                BartDecoderLayer(
                    config.d_model,
                    config.decoder_attention_heads,
                    config.decoder_ffn_dim,
                    config.dropout,
                    config.activation_dropout,
                    config.activation_function,
                )
                for _ in range(config.decoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.dropout = config.dropout

    def forward(self, input_ids, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(input_ids.shape[1], device=input_ids.device)
        hidden_states = self.layernorm_embedding(inputs_embeds + embed_pos)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        seq_len = input_ids.shape[1]
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, device=input_ids.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None]
        encoder_mask = _expand_mask(encoder_attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, encoder_hidden_states, attention_mask=causal_mask, encoder_attention_mask=encoder_mask
            )
        return hidden_states


class BartModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

    def forward(self, input_ids, decoder_input_ids, attention_mask=None, encoder_attention_mask=None):
        encoder_hidden_states = self.encoder(input_ids, attention_mask=attention_mask)
        decoder_hidden_states = self.decoder(
            decoder_input_ids, encoder_hidden_states, encoder_attention_mask=encoder_attention_mask
        )
        return encoder_hidden_states, decoder_hidden_states


class BartForConditionalGeneration(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.shared.weight
        self.register_buffer("final_logits_bias", torch.zeros(1, config.vocab_size))

    def forward(self, input_ids, decoder_input_ids, attention_mask=None, encoder_attention_mask=None):
        _, decoder_hidden_states = self.model(input_ids, decoder_input_ids, attention_mask, encoder_attention_mask)
        return self.lm_head(decoder_hidden_states) + self.final_logits_bias

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_length,
        attention_mask=None,
        decoder_start_token_id=None,
        eos_token_id=None,
        forced_eos_token_id=None,
    ):
        decoder_start = self.config.decoder_start_token_id if decoder_start_token_id is None else decoder_start_token_id
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        forced_eos = self.config.forced_eos_token_id if forced_eos_token_id is None else forced_eos_token_id
        decoder_input_ids = torch.full((input_ids.shape[0], 1), decoder_start, dtype=torch.long, device=input_ids.device)
        encoder_hidden_states = self.model.encoder(input_ids, attention_mask=attention_mask)
        encoder_mask = attention_mask
        for _ in range(max_length - 1):
            decoder_hidden_states = self.model.decoder(
                decoder_input_ids, encoder_hidden_states, encoder_attention_mask=encoder_mask
            )
            logits = self.lm_head(decoder_hidden_states[:, -1]) + self.final_logits_bias
            next_token = logits.argmax(dim=-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token[:, None]], dim=1)
            if (next_token == eos).all():
                break
        if forced_eos is not None and decoder_input_ids.shape[1] >= max_length:
            decoder_input_ids[:, -1] = forced_eos
        return decoder_input_ids
