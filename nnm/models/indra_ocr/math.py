import math

import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.sinusoidal import sinusoidal_embedding
from nnm.models.indra_ocr.rec import ConvBlock
from nnm.models.pretrained import PretrainedModel


class MathClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample = config.downsample
        self.pool = nn.AvgPool2d(2, 2, count_include_pad=True)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
                for in_channels, out_channels in zip([1, *config.channels[:-1]], config.channels)
            ]
        )
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in config.channels])
        self.head = nn.Linear(config.channels[-1], 1)

    def forward(self, x, widths):
        hidden = self.pool(x)
        for conv, pool in zip(self.convs, self.pools):
            hidden = pool(F.relu(conv(hidden)))
        hidden = hidden.squeeze(2)
        steps = (widths // self.downsample).clamp(min=1)
        mask = (torch.arange(hidden.shape[-1], device=x.device) < steps[:, None]).to(hidden.dtype).unsqueeze(1)
        hidden = (hidden * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return self.head(hidden).squeeze(-1).sigmoid()


class MathEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.front = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
                for in_channels, out_channels in zip([1, *config.front_channels[:-1]], config.front_channels)
            ]
        )
        self.fold = nn.Conv2d(config.front_channels[-1], config.channels, (config.input_height // 8, 1))
        self.fold_norm = nn.BatchNorm1d(config.channels)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(config.channels, config.expand_channels, config.reduced_channels, config.kernel_size)
                for _ in range(config.blocks)
            ]
        )
        self.layers = nn.ModuleList([MathEncoderLayer(config) for _ in range(config.layers)])
        self.head_dim = config.channels // config.heads

    def positional_encoding(self, length, dtype):
        positions = torch.arange(length, device=self.fold_norm.weight.device)
        encoding = sinusoidal_embedding(positions, self.config.channels, self.config.position_base, interleaved=True)
        return encoding.to(dtype).unsqueeze(0)

    def forward(self, x, widths):
        hidden = x
        for conv in self.front:
            hidden = F.gelu(conv(hidden))
        hidden = self.fold(hidden).squeeze(2)
        hidden = F.gelu(self.fold_norm(hidden))

        steps = (widths + self.config.downsample - 1) // self.config.downsample
        steps = steps.clamp(min=1, max=hidden.shape[-1])
        mask = (torch.arange(hidden.shape[-1], device=x.device) < steps[:, None]).to(hidden.dtype).unsqueeze(1)
        count = mask.sum(-1, keepdim=True).clamp(min=1)

        hidden = hidden * mask
        for block in self.blocks:
            hidden = block(hidden, mask, count)

        hidden = hidden.transpose(1, 2)
        for layer in self.layers:
            hidden = layer(hidden, mask.squeeze(1))
        hidden = hidden * mask.squeeze(1).unsqueeze(-1)

        memory = hidden + self.positional_encoding(hidden.shape[1], hidden.dtype)
        return memory, mask.squeeze(1)


class MathEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.channels // config.heads
        self.register_buffer(
            "head_penalty",
            torch.tensor([0.5 ** (i + 1) for i in range(config.heads)], dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.norm1 = nn.LayerNorm(config.channels, eps=1e-5)
        self.qkv = nn.Linear(config.channels, 3 * config.channels)
        self.out = nn.Linear(config.channels, config.channels)
        self.norm2 = nn.LayerNorm(config.channels, eps=1e-5)
        self.ff1 = nn.Linear(config.channels, config.ffn_channels)
        self.ff2 = nn.Linear(config.ffn_channels, config.channels)

    def forward(self, x, mask):
        residual = x
        hidden = self.norm1(x)
        qkv = self.qkv(hidden).reshape(*hidden.shape[:2], 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        distance = (
            torch.arange(x.shape[1], device=x.device)[:, None] - torch.arange(x.shape[1], device=x.device)[None, :]
        ).abs()
        scores = scores - distance.to(scores.dtype) * self.head_penalty
        scores = scores + (mask - 1).to(scores.dtype)[:, None, None, :] * 10000
        hidden = torch.softmax(scores, dim=-1) @ value
        hidden = hidden.transpose(1, 2).reshape(*x.shape[:2], self.heads * self.head_dim)
        x = residual + self.out(hidden)

        residual = x
        hidden = self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return residual + hidden


class MathDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.max_tokens + 1, config.d_model)
        self.layers = nn.ModuleList([MathDecoderLayer(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def encode_memory(self, memory):
        keys, values = [], []
        for layer in self.layers:
            key, value = layer.cross_project(memory)
            keys.append(key)
            values.append(value)
        return torch.stack(keys), torch.stack(values)

    def forward(self, token, position, self_key, self_value, cross_key, cross_value, mask):
        hidden = self.embedding(token).unsqueeze(1) + self.position(position)
        new_keys, new_values = [], []
        for index, layer in enumerate(self.layers):
            hidden, key, value = layer.self_attention(hidden, self_key[index], self_value[index])
            new_keys.append(key)
            new_values.append(value)
            hidden = layer.cross_attention(hidden, cross_key[index], cross_value[index], mask)
            hidden = layer.feed_forward(hidden)
        logits = self.head(self.norm(hidden)).squeeze(1)
        return logits, torch.stack(new_keys), torch.stack(new_values)

    def empty_cache(self, batch, device, dtype):
        shape = (self.config.layers, batch, self.config.heads, 0, self.config.head_dim)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)


class MathDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.scale = math.sqrt(config.head_dim)
        self.self_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.self_qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.self_out = nn.Linear(config.d_model, config.d_model)
        self.cross_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.cross_qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.cross_out = nn.Linear(config.d_model, config.d_model)
        self.ff_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.ff1 = nn.Linear(config.d_model, config.ffn_channels)
        self.ff2 = nn.Linear(config.ffn_channels, config.d_model)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.d_model = config.d_model

    def split(self, x, linear):
        hidden = linear(x).reshape(*x.shape[:-1], 3, self.heads, self.head_dim)
        return hidden.permute(2, 0, 3, 1, 4).unbind(0)

    def self_attention(self, x, cached_key, cached_value):
        residual = x
        query, key, value = self.split(self.self_norm(x), self.self_qkv)
        key = torch.cat([cached_key, key], dim=2)
        value = torch.cat([cached_value, value], dim=2)
        scores = query @ key.transpose(-1, -2) / self.scale
        hidden = torch.softmax(scores, dim=-1) @ value
        hidden = hidden.transpose(1, 2).reshape(*x.shape[:-1], self.heads * self.head_dim)
        return residual + self.dropout(self.self_out(hidden)), key, value

    def cross_project(self, memory):
        _, key, value = self.split(memory, self.cross_qkv)
        return key, value

    def cross_attention(self, x, key, value, mask):
        residual = x
        query, _, _ = self.split(self.cross_norm(x), self.cross_qkv)
        scores = query @ key.transpose(-1, -2) / self.scale
        scores = scores + (mask - 1).to(scores.dtype)[:, None, None, :] * 1e9
        hidden = torch.softmax(scores, dim=-1) @ value
        hidden = hidden.transpose(1, 2).reshape(*x.shape[:-1], self.heads * self.head_dim)
        return residual + self.dropout(self.cross_out(hidden))

    def feed_forward(self, x):
        return x + self.dropout(self.ff2(F.gelu(self.ff1(self.ff_norm(x)))))


class IndraOCRMath(PretrainedModel):
    module_name = "math"

    BOS = 1
    EOS = 2
    N_SPECIAL = 4
    MIN_CONF = 0.9
    EARLY_STOP_MARGIN = 0.1

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = MathClassifier(config.classifier)
        self.encoder = MathEncoder(config.encoder)
        self.decoder = MathDecoder(config.decoder)
        self.tokens = ["", "", "", "", *config.tokens]

    def forward(self, x, widths):
        return self.classifier(x, widths)

    def encode(self, x, widths):
        return self.encoder(x, widths)

    def flags(self, x, widths):
        return self.classifier(x[:, :, : self.config.classifier.input_height], widths) > 0.5

    def detokenize(self, tokens):
        out, math, inside = [], [], False
        for token in tokens:
            if token == "$":
                if inside:
                    out.append("$" + " ".join(math) + "$")
                    math = []
                inside = not inside
            elif inside:
                math.append(token)
            else:
                out.append(token)
        if inside and math:
            out.append("$" + " ".join(math) + "$")
        return " ".join("".join(out).split())

    def decode(self, x, widths):
        memory, mask = self.encode(x, widths)
        cross_key, cross_value = self.decoder.encode_memory(memory)
        self_key, self_value = self.decoder.empty_cache(x.shape[0], x.device, memory.dtype)

        tokens = torch.full((x.shape[0],), self.BOS, dtype=torch.long, device=x.device)
        sequences = [[] for _ in range(x.shape[0])]
        probabilities = [[] for _ in range(x.shape[0])]
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for step in range(self.config.decoder.max_tokens - 1):
            logits, self_key, self_value = self.decoder(
                tokens, torch.tensor(step, device=x.device), self_key, self_value, cross_key, cross_value, mask
            )
            tokens = logits.argmax(-1)
            confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
            for index in range(x.shape[0]):
                if done[index]:
                    continue
                probabilities[index].append(float(confidence[index]))
                token = int(tokens[index])
                mean = sum(probabilities[index]) / len(probabilities[index])
                early = len(probabilities[index]) >= 8 and mean < self.MIN_CONF - self.EARLY_STOP_MARGIN
                if token == self.EOS or early:
                    done[index] = True
                elif token >= self.N_SPECIAL:
                    sequences[index].append(token)
            if done.all():
                break

        return [
            (
                self.detokenize([self.tokens[token] for token in sequence]),
                float(sum(probability) / len(probability)) if probability else 0.0,
            )
            for sequence, probability in zip(sequences, probabilities)
        ]
