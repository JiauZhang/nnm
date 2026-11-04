import torch
from torch import nn


class VisualVocabularyHead(nn.Module):
    def __init__(self, *, hidden_size, hidden_stride, vocab_size, indicator_count):
        super().__init__()
        head_dim = vocab_size - indicator_count
        self.proj = nn.Linear(hidden_size * hidden_stride**2, head_dim, bias=False)
        self.norm = nn.LayerNorm(head_dim)

    def forward(self, features):
        return self.norm(self.proj(features))


class VisualVocabularyEmbedding(nn.Embedding):
    def forward(self, visual_tokens):
        if torch.is_floating_point(visual_tokens):
            return torch.matmul(visual_tokens, self.weight)
        return super().forward(visual_tokens)


class OXRVisualTokenizer(nn.Module):
    def __init__(self, *, vit, visual_vocab_size, indicator_count):
        super().__init__()
        self.vit = vit
        self.indicator_count = indicator_count
        self.head = VisualVocabularyHead(
            hidden_size=vit.embeddings.embed_dim,
            hidden_stride=vit.embeddings.hidden_stride,
            vocab_size=visual_vocab_size,
            indicator_count=indicator_count,
        )

    def encode(self, pixel_values, grid_thws):
        _, hidden_states = self.vit(pixel_values, grid_thws, output_hidden_states=True)
        features = hidden_states[-1]
        merge_unit = self.vit.embeddings.hidden_stride**2
        return features.reshape(features.shape[0] // merge_unit, -1)

    def forward(self, pixel_values, grid_thws):
        logits = self.head(self.encode(pixel_values, grid_thws))
        tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        padding = torch.zeros(tokens.shape[0], self.indicator_count, dtype=tokens.dtype, device=tokens.device)
        return torch.cat((tokens, padding), dim=-1)
