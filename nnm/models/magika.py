import torch
import torch.nn as nn
import torch.nn.functional as F


class Magika(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.token_dim)
        self.seq_norm = nn.LayerNorm(config.seq_len // 4, eps=1e-6)
        self.seq_conv = nn.Conv1d(
            in_channels=config.token_dim * 4,
            out_channels=config.hidden_dim,
            kernel_size=config.conv_kernel,
            padding=config.conv_kernel // 2,
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.feature_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, bytes):
        assert bytes.shape[1] == self.config.seq_len, f"Input byte tensor must have shape [batch, {self.config.seq_len}]"

        embeddings = self.token_embed(bytes.long())
        hidden_states = F.gelu(embeddings, approximate="tanh")
        hidden_states = hidden_states.reshape(-1, self.config.seq_len // 4, self.config.token_dim * 4)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.seq_norm(hidden_states)
        hidden_states = self.seq_conv(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        pooled = self.pool(hidden_states).squeeze(-1)
        hidden_states = self.feature_norm(pooled)
        logits = self.head(hidden_states)
        probs = F.softmax(logits, dim=-1)

        return probs
