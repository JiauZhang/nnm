import torch
from torch import nn

from .blocks import ConvLayerNormAct1d, StyleLSTMStage


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.text_encoder
        slope = config.predictor.leaky_relu_slope
        self.embedding = nn.Embedding(config.albert.vocab_size, c.embed_dim)
        self.cnn = nn.Sequential(
            ConvLayerNormAct1d(c.embed_dim, c.cnn_hidden, c.cnn_kernel_size, c.cnn_padding, slope),
            ConvLayerNormAct1d(c.cnn_hidden, c.cnn_hidden, c.cnn_kernel_size, c.cnn_padding, slope),
        )
        self.cnn_lstm = nn.LSTM(c.cnn_hidden, c.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.lstm_stages = nn.ModuleList(
            [StyleLSTMStage(c.lstm_chain_input, c.lstm_hidden_size, c.adain_style_dim, slope) for _ in range(2)]
        )

    def forward(self, input_ids, style, bert_output=None):
        batch_size, seq_len = input_ids.shape
        style_for_lstm = style[:, style.shape[-1] // 2 :]

        cnn_embed = self.embedding(input_ids)
        cnn_out = self.cnn(cnn_embed.transpose(1, 2)).transpose(1, 2)

        cnn_lstm_out, _ = self.cnn_lstm(cnn_out)
        cnn_features = cnn_lstm_out.transpose(1, 2)

        if bert_output is None:
            bert_output = cnn_out

        hidden = bert_output
        for stage in self.lstm_stages:
            hidden = stage(hidden, style_for_lstm)

        style_broadcast = style_for_lstm.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_features = torch.cat([hidden, style_broadcast], dim=-1).transpose(1, 2)

        return lstm_features, cnn_features
