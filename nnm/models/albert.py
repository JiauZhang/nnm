import torch
import torch.nn as nn

from nnm.layers.attention import SelfAttention
from nnm.layers.mlp import MLP


class AlbertEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_position_embeddings, type_vocab_size, layer_norm_eps):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.0)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        token_types = self.token_type_embeddings(torch.zeros_like(input_ids))

        embeddings = words + token_types + positions
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertLayer(nn.Module):
    def __init__(
        self, hidden_size, num_heads, intermediate_size, layer_norm_eps, head_dim=None, activation="gelu_pytorch_tanh"
    ):
        super().__init__()
        self.attention = SelfAttention(
            hidden_size,
            num_heads,
            head_dim=head_dim,
            qkv_bias=True,
            o_bias=True,
            causal=False,
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn = MLP(hidden_size, intermediate_size, activation=activation, bias=True)
        self.full_layer_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention_layer_norm(hidden_states + self.attention(hidden_states, attn_mask=attention_mask))
        ffn_output = self.ffn(attention_output)
        return self.full_layer_layer_norm(ffn_output + attention_output)


class AlbertLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert_layers = nn.ModuleList(
            [
                AlbertLayer(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                    config.layer_norm_eps,
                    activation=config.hidden_act,
                )
                for _ in range(config.inner_group_num)
            ]
        )

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.albert_layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class AlbertEncoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        layers_per_group = self.num_hidden_layers // self.num_hidden_groups
        for group in self.albert_layer_groups:
            for _ in range(layers_per_group):
                hidden_states = group(hidden_states, attention_mask)
        return hidden_states


class AlbertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = AlbertEmbeddings(
            config.vocab_size,
            config.embedding_size,
            config.max_position_embeddings,
            config.type_vocab_size,
            config.layer_norm_eps,
        )
        self.encoder = AlbertEncoderStack(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_mask = (1.0 - attention_mask[:, None, None, :].to(embedding_output.dtype)) * torch.finfo(
            embedding_output.dtype
        ).min
        return self.encoder(embedding_output, extended_mask)
