import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AlbertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = config.head_dim
        self.all_head_size = config.num_heads * config.head_dim

        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)
        self.dense = nn.Linear(self.all_head_size, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)
        self.dropout = nn.Dropout(0.0)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        projected_context_layer = self.dense(context_layer)
        projected_context_layer = self.dropout(projected_context_layer)
        layernormed_context_layer = self.layer_norm(hidden_states + projected_context_layer)

        return layernormed_context_layer


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_dim, config.ffn_dim)
        self.ffn_output = nn.Linear(config.ffn_dim, config.hidden_dim)
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)
        self.dropout = nn.Dropout(0.0)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(attention_output)
        ffn_output = F.gelu(ffn_output, approximate='tanh')
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.dropout(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        return hidden_states


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
