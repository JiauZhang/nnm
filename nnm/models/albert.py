import torch
import torch.nn as nn

from nnm.layers.attention import SelfAttention
from nnm.layers.mlp import MLP


class AlbertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.token_type_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_position, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.layernorm_eps)
        self.dropout = nn.Dropout(0.0)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        token_types = self.token_type_embeddings(torch.zeros_like(input_ids))

        embeddings = words + positions + token_types
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(
            config.hidden_dim,
            config.num_heads,
            head_dim=config.head_dim,
            qkv_bias=True,
            o_bias=True,
            causal=False,
        )
        self.attention_layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)
        self.ffn = MLP(config.hidden_dim, config.ffn_dim, activation="gelu_pytorch_tanh", bias=True)
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention_layer_norm(hidden_states + self.attention(hidden_states, attn_mask=attention_mask))
        ffn_output = self.ffn(attention_output)
        return self.full_layer_layer_norm(ffn_output + attention_output)


class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = AlbertEmbeddings(config)
        self.embedding_hidden_mapping_in = nn.Linear(config.embed_dim, config.hidden_dim)
        self.shared_layer = AlbertLayer(config)
        self.num_layers = config.num_layers
        self.bert_encoder = nn.Linear(config.hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids)
        hidden_states = self.embedding_hidden_mapping_in(embedding_output)

        for _ in range(self.num_layers):
            hidden_states = self.shared_layer(hidden_states, attention_mask)

        output = self.bert_encoder(hidden_states)
        output = self.dropout(output)
        return output
