import torch
from torch import nn
from torch.nn.utils import weight_norm

from .blocks import ChannelLayerNorm


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                    ChannelLayerNorm(channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                )
            )
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, mask):
        x = self.embedding(x).transpose(1, 2)
        mask = mask.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(mask, 0.0)
        for block in self.cnn:
            x = block(x)
            x.masked_fill_(mask, 0.0)
        x = x.transpose(1, 2)

        lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.transpose(-1, -2)
        x_pad = torch.zeros(x.shape[0], x.shape[1], mask.shape[-1], device=x.device, dtype=x.dtype)
        x_pad[:, :, : x.shape[-1]] = x
        x_pad.masked_fill_(mask, 0.0)
        return x_pad
