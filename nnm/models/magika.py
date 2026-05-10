import torch
from torch import nn
from nnm.models.pretrained import PretrainedModel


class Magika(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.token_dim = config.token_dim
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes
        self.conv_kernel = config.conv_kernel
        self.expand_scale = config.expand_scale

        self.embedding = nn.Embedding(config.vocab_size, self.token_dim)
        self.embed_bias = nn.Parameter(torch.zeros(self.token_dim))
        self.activation_0 = nn.GELU(approximate='tanh')
        self.layer_norm_0 = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.conv_0 = nn.Conv1d(self.token_dim * self.expand_scale, self.hidden_dim, kernel_size=self.conv_kernel, padding=0)
        self.activation_1 = nn.GELU(approximate='tanh')
        self.layer_norm_1 = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.dense_1 = nn.Linear(self.hidden_dim, self.num_classes)
        self.whitespace = torch.tensor([ord(c) for c in '\t\n\r '])

    @property
    def labels(self):
        return self.config.labels

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        beg_size = self.config.feat_beg_size
        end_size = self.config.feat_end_size
        pad_token = self.config.feat_padding_token
        block_size = self.config.feat_block_size

        x = x.flatten()
        x = x[:block_size]
        mask = ~torch.isin(x, self.whitespace.to(device=x.device))
        non_ws = torch.where(mask)[0]
        mask_any = mask.any()

        beg = self._trim_segment(x, non_ws, mask_any, beg_size, pad_token, 'left')
        end = self._trim_segment(x, non_ws, mask_any, end_size, pad_token, 'right')
        return torch.cat([beg, end]).reshape(1, -1)

    def _trim_segment(self, x, non_ws, mask_any, size, pad_token, direction):
        if not mask_any:
            result = torch.tensor([], dtype=torch.int64, device=x.device)
        elif direction == 'left':
            result = x[non_ws[0]:]
        else:
            result = x[:non_ws[-1] + 1]
        if result.size(0) > size:
            result = result[:size] if direction == 'left' else result[-size:]
        if result.size(0) < size:
            pad = torch.full((size - result.size(0),), pad_token, dtype=torch.int64, device=x.device)
            result = torch.cat([result, pad]) if direction == 'left' else torch.cat([pad, result])
        return result

    def forward(self, x):
        x = self.extract_features(x).to(torch.int64)
        x = self.embedding(x)
        x = x + self.embed_bias
        x = self.activation_0(x)
        B = x.shape[0]
        x = x.reshape(B, self.seq_len // self.expand_scale, self.token_dim * self.expand_scale)
        x = x.transpose(1, 2)
        x = self.layer_norm_0(x)
        x = self.conv_0(x)
        x = self.activation_1(x)
        x = x.max(dim=-1)[0]
        x = self.layer_norm_1(x)
        x = self.dense_1(x)
        probs = torch.softmax(x, dim=-1)
        return probs