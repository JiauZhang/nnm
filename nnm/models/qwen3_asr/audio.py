import torch
import torch.nn.functional as F
from functools import partial
from torch import nn

from nnm.layers.activation import get_activation
from nnm.layers.attention import SelfAttention
from nnm.layers.mlp import MLP
from nnm.layers.sinusoidal import SinusoidalPositionEmbedding
from nnm.layers.transformer import TransformerBlock


def feat_extract_lengths(input_lengths):
    leave = input_lengths % 100
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class AudioAttention(SelfAttention):
    def __init__(self, config):
        embed_dim = config.d_model
        num_heads = config.encoder_attention_heads
        super().__init__(embed_dim, num_heads, num_heads, embed_dim // num_heads, qkv_bias=True, o_bias=True, causal=False)

    def forward(self, hidden_states, cu_seqlens):
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return super().forward(hidden_states[None], lengths=lengths)


def audio_encoder_layer(config):
    return TransformerBlock(
        config.d_model,
        AudioAttention(config),
        MLP(config.d_model, config.encoder_ffn_dim, "gelu", bias=True),
        norm_layer=partial(nn.LayerNorm, eps=1e-5),
    )


class Qwen3ASRAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        self.positional_embedding = SinusoidalPositionEmbedding(
            self.max_source_positions, config.d_model, denominator=config.d_model // 2 - 1
        )
        self.layers = nn.ModuleList([audio_encoder_layer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = get_activation(config.activation_function)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def forward(self, input_features, feature_lens):
        aftercnn_lens = feat_extract_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor([self.n_window * 2] * chunk_num.sum(), dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2).unsqueeze(1)
        feature_lens_after_cnn = feat_extract_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )

        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            embed = F.gelu(self.conv2d1(chunk))
            embed = F.gelu(self.conv2d2(embed))
            padded_embeds.append(F.gelu(self.conv2d3(embed)))
        padded_embed = torch.cat(padded_embeds, dim=0)
        batch, channels, freq, time = padded_embed.shape
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).reshape(batch, time, channels * freq))

        positional_embedding = self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        padded_embed = padded_embed + positional_embedding[None].to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]

        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens.tolist():
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            if cnn_len % window_aftercnn != 0:
                cu_chunk_lens.append(cnn_len % window_aftercnn)
        cu_seqlens = torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens)
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        hidden_states = self.proj1(self.ln_post(hidden_states))
        return self.proj2(F.gelu(hidden_states))
