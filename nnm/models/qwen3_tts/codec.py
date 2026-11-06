import math

import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.attention import RotarySelfAttention
from nnm.layers.mlp import GatedMLP
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.vit import LayerScale
from nnm.layers.vq import ResidualVectorQuantization
from nnm.models.pretrained import PretrainedModel


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def extra_padding(self, length):
        frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal = (math.ceil(frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal - length

    def forward(self, hidden):
        hidden = F.pad(hidden, (self.padding, self.extra_padding(hidden.shape[-1])), mode="constant", value=0)
        return self.conv(hidden).contiguous()


class CausalTransConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.right_pad = int(kernel_size - stride)

    def forward(self, hidden):
        hidden = self.conv(hidden)
        if self.right_pad > 0:
            hidden = hidden[..., : hidden.shape[-1] - self.right_pad]
        return hidden.contiguous()


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = CausalConv(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden):
        residual = hidden
        hidden = self.dwconv(hidden).permute(0, 2, 1)
        hidden = self.pwconv2(self.act(self.pwconv1(self.norm(hidden))))
        hidden = self.gamma * hidden
        return residual + hidden.permute(0, 2, 1)


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(hidden * alpha), 2)


class DecoderResidualUnit(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv(dim, dim, kernel_size=1)

    def forward(self, hidden):
        residual = hidden
        hidden = self.conv2(self.act2(self.conv1(self.act1(hidden))))
        return hidden + residual


class DecoderBlock(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_index
        out_dim = config.decoder_dim // 2 ** (layer_index + 1)
        rate = config.upsample_rates[layer_index]
        block = [SnakeBeta(in_dim), CausalTransConv(in_dim, out_dim, 2 * rate, rate)]
        for dilation in (1, 3, 9):
            block.append(DecoderResidualUnit(out_dim, dilation))
        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden


class CodecAttention(RotarySelfAttention):
    def __init__(self, config):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        super().__init__(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            qkv_bias=config.attention_bias,
            o_bias=config.attention_bias,
            sliding_window=config.sliding_window,
        )


class CodecLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CodecAttention(config)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = LayerScale(config.hidden_size, config.layer_scale_initial_scale)
        self.mlp_layer_scale = LayerScale(config.hidden_size, config.layer_scale_initial_scale)

    def forward(self, hidden, rotary_embeddings):
        residual = hidden
        hidden = residual + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(hidden), rotary_embeddings))
        residual = hidden
        return residual + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(hidden)))


class CodecTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)
        self.layers = nn.ModuleList([CodecLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = RoPE(rotary_inv_freq(head_dim, config.rope_theta))

    def forward(self, hidden):
        hidden = self.input_proj(hidden)
        positions = torch.arange(0, hidden.shape[1], device=hidden.device)
        rotary_embeddings = self.rotary_emb(positions)
        for layer in self.layers:
            hidden = layer(hidden, rotary_embeddings)
        return self.output_proj(self.norm(hidden))


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dimension, input_dimension, output_dimension, n_q, bins):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dimension, dimension, 1, bias=False)
        self.output_proj = nn.Conv1d(dimension, output_dimension, 1, bias=False)
        self.vq = ResidualVectorQuantization(dim=dimension, codebook_size=bins, num_quantizers=n_q)

    def decode(self, codes):
        codes = codes.transpose(0, 1)
        return self.output_proj(self.vq.decode(codes))


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, dimension, num_quantizers, codebook_size, input_dimension, output_dimension):
        super().__init__()
        self.n_q_semantic = 1
        self.rvq_first = ResidualVectorQuantizer(dimension, input_dimension, output_dimension, 1, codebook_size)
        self.rvq_rest = ResidualVectorQuantizer(dimension, input_dimension, output_dimension, num_quantizers - 1, codebook_size)

    def decode(self, codes):
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class CodecDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.total_upsample = math.prod(config.upsample_rates + config.upsampling_ratios)
        self.pre_transformer = CodecTransformer(config)
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )
        self.pre_conv = CausalConv(config.codebook_dim, config.latent_dim, kernel_size=3)
        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [CausalTransConv(config.latent_dim, config.latent_dim, factor, factor), ConvNeXtBlock(config.latent_dim)]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [CausalConv(config.latent_dim, config.decoder_dim, 7)]
        for index in range(len(config.upsample_rates)):
            decoder.append(DecoderBlock(config, index))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [SnakeBeta(output_dim), CausalConv(output_dim, 1, 7)]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        for block in self.decoder:
            hidden = block(hidden)
        return hidden.clamp(min=-1, max=1)


class Qwen3TTSCodec(PretrainedModel):
    module_name = "codec"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = CodecDecoder(config.decoder_config)

    def decode(self, codes):
        if codes.dim() == 2:
            codes = codes.transpose(0, 1).unsqueeze(0)
        if codes.shape[1] != self.config.decoder_config.num_quantizers:
            raise ValueError(f"Expected {self.config.decoder_config.num_quantizers} codebooks, got {codes.shape[1]}")
        return self.decoder(codes)
