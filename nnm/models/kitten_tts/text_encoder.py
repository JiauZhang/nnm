import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        normalized = normalized * self.weight.view(1, -1, 1)
        normalized = normalized + self.bias.view(1, -1, 1)
        return normalized


def _reorder_lstm_weights_iofc_to_ifco(W, R, B):
    hidden = R.shape[0] // 4

    idx = torch.cat([
        torch.arange(0, hidden),
        torch.arange(2 * hidden, 3 * hidden),
        torch.arange(3 * hidden, 4 * hidden),
        torch.arange(hidden, 2 * hidden),
    ])

    W_ifco = W[idx]
    R_ifco = R[idx]

    B_ih = B[: len(B) // 2]
    B_hh = B[len(B) // 2 :]

    return W_ifco, R_ifco, B_ih[idx], B_hh[idx]


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_fwd = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.R_fwd = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.B_fwd = nn.Parameter(torch.empty(8 * hidden_size))
        self.W_rev = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.R_rev = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.B_rev = nn.Parameter(torch.empty(8 * hidden_size))

        self._lstm = None
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.W_fwd, self.W_rev]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for r in [self.R_fwd, self.R_rev]:
            nn.init.orthogonal_(r)
        for b in [self.B_fwd, self.B_rev]:
            nn.init.zeros_(b)

    def _get_native_lstm(self):
        if self._lstm is not None:
            return self._lstm

        input_size = self.W_fwd.shape[1]
        hidden_size = self.hidden_size

        self._lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)

        W_f, R_f, B_ih_f, B_hh_f = _reorder_lstm_weights_iofc_to_ifco(
            self.W_fwd, self.R_fwd, self.B_fwd
        )
        self._lstm.weight_ih_l0.data = W_f
        self._lstm.weight_hh_l0.data = R_f
        self._lstm.bias_ih_l0.data = B_ih_f
        self._lstm.bias_hh_l0.data = B_hh_f

        W_r, R_r, B_ih_r, B_hh_r = _reorder_lstm_weights_iofc_to_ifco(
            self.W_rev, self.R_rev, self.B_rev
        )
        self._lstm.weight_ih_l0_reverse.data = W_r
        self._lstm.weight_hh_l0_reverse.data = R_r
        self._lstm.bias_ih_l0_reverse.data = B_ih_r
        self._lstm.bias_hh_l0_reverse.data = B_hh_r

        return self._lstm

    def forward(self, X):
        lstm = self._get_native_lstm()
        output, _ = lstm(X)
        return output


class AdaIN(nn.Module):
    def __init__(self, style_in, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.fc = nn.Linear(style_in, 2 * feat_dim)

    def forward(self, x, style):
        transpose_back = x.dim() == 3 and x.shape[0] == style.shape[0]
        if transpose_back:
            x = x.transpose(0, 1)

        seq_len, batch_size, feat_dim = x.shape
        x_norm = F.layer_norm(x, [feat_dim])

        proj = self.fc(style)
        scale = proj[:, :feat_dim] + 1.0
        bias = proj[:, feat_dim:]

        out = x_norm * scale.unsqueeze(0) + bias.unsqueeze(0)

        if transpose_back:
            out = out.transpose(0, 1)

        return out


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.text_encoder
        self.embedding = nn.Embedding(config.albert.vocab_size, c.embed_dim)
        self.cnn_conv1 = nn.Conv1d(c.embed_dim, c.cnn_hidden, kernel_size=c.cnn_kernel_size, padding=c.cnn_padding)
        self.cnn_norm1 = LayerNorm1d(c.cnn_hidden)
        self.cnn_conv2 = nn.Conv1d(c.cnn_hidden, c.cnn_hidden, kernel_size=c.cnn_kernel_size, padding=c.cnn_padding)
        self.cnn_norm2 = LayerNorm1d(c.cnn_hidden)
        self.leaky_relu = nn.LeakyReLU(config.predictor.leaky_relu_slope)

        self.cnn_lstm = BidirectionalLSTM(input_size=c.cnn_hidden, hidden_size=c.lstm_hidden_size)

        self.lstm0 = BidirectionalLSTM(input_size=c.lstm_chain_input, hidden_size=c.lstm_hidden_size)
        self.adain1 = AdaIN(c.adain_style_dim, c.lstm_hidden_size * 2)
        self.lstm2 = BidirectionalLSTM(input_size=c.lstm_chain_input, hidden_size=c.lstm_hidden_size)
        self.adain3 = AdaIN(c.adain_style_dim, c.lstm_hidden_size * 2)

    def forward(self, input_ids, style, bert_output=None):
        batch_size, seq_len = input_ids.shape
        style_for_lstm = style[:, style.shape[-1] // 2 :]

        cnn_embed = self.embedding(input_ids)
        cnn_in = cnn_embed.transpose(1, 2)
        cnn_out = self.cnn_conv1(cnn_in)
        cnn_out = self.cnn_norm1(cnn_out)
        cnn_out = self.leaky_relu(cnn_out)
        cnn_out = self.cnn_conv2(cnn_out)
        cnn_out = self.cnn_norm2(cnn_out)
        cnn_out = self.leaky_relu(cnn_out)
        cnn_out = cnn_out.transpose(1, 2)

        cnn_lstm_in = cnn_out.transpose(0, 1)
        cnn_lstm_out = self.cnn_lstm(cnn_lstm_in)
        cnn_features = cnn_lstm_out.transpose(0, 1).transpose(1, 2)

        if bert_output is None:
            bert_output = cnn_out

        bert_t = bert_output.transpose(0, 1)

        style_broadcast = style_for_lstm.unsqueeze(0).expand(seq_len, -1, -1)
        lstm0_in = torch.cat([bert_t, style_broadcast], dim=-1)
        lstm0_out = self.lstm0(lstm0_in)

        lstm0_normed = self.adain1(lstm0_out, style_for_lstm)

        style_broadcast2 = style_for_lstm.unsqueeze(0).expand(seq_len, -1, -1)
        lstm2_in = torch.cat([lstm0_normed, style_broadcast2], dim=-1)
        lstm2_out = self.lstm2(lstm2_in)

        lstm2_normed = self.adain3(lstm2_out, style_for_lstm)

        style_broadcast3 = style_for_lstm.unsqueeze(0).expand(seq_len, -1, -1)
        lstm_features_seq = torch.cat([lstm2_normed, style_broadcast3], dim=-1)
        lstm_features = lstm_features_seq.transpose(0, 1).transpose(1, 2)

        return lstm_features, cnn_features
