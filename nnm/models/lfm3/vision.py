import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.activation import get_activation
from nnm.layers.mlp import SwiGLU
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE2D
from nnm.layers.transformer import SelfAttention


def vision_inv_freq(config):
    head_dim = config.hidden_size // config.num_attention_heads
    spatial_dim = head_dim // 2
    base = config.rope_parameters["rope_theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
    return [inv_freq, inv_freq.clone()]


def vision_position_ids(grid_thw, spatial_merge_size):
    device = grid_thw.device
    positions = []
    for frames, height, width in grid_thw.tolist():
        height_ids = torch.arange(height, device=device).view(height, 1).expand(height, width)
        width_ids = torch.arange(width, device=device).view(1, width).expand(height, width)
        block = (height // spatial_merge_size, spatial_merge_size, width // spatial_merge_size, spatial_merge_size)
        height_ids = height_ids.reshape(block).transpose(1, 2).flatten()
        width_ids = width_ids.reshape(block).transpose(1, 2).flatten()
        positions.append(torch.stack([height_ids, width_ids], dim=-1).repeat(frames, 1))
    return torch.cat(positions, dim=0)


def vision_frame_shapes(grid_thw):
    return grid_thw[:, 1:].repeat_interleave(grid_thw[:, 0], dim=0).tolist()


def vision_cu_seqlens(grid_thw):
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0)


class Lfm3VisionAttention(SelfAttention):
    def __init__(self, config):
        head_dim = config.hidden_size // config.num_attention_heads
        super().__init__(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            q_norm=True,
            k_norm=True,
            norm_eps=config.norm_eps,
            causal=False,
        )

    def forward(self, x, lengths, cos, sin):
        return super().forward(x[None], cos=cos, sin=sin, lengths=lengths)


class Lfm3VisionConv2d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, frame_shapes):
        gate, scale, value = self.in_proj(x).chunk(3, dim=-1)
        gated = gate * value
        merge = self.spatial_merge_size
        if len(set(map(tuple, frame_shapes))) == 1:
            height, width = frame_shapes[0]
            frames = gated.view(-1, height // merge, width // merge, merge, merge, self.hidden_size)
            frames = frames.permute(0, 5, 1, 3, 2, 4).reshape(-1, self.hidden_size, height, width)
            frames = self.spatial_conv(frames)
            frames = frames.view(-1, self.hidden_size, height // merge, merge, width // merge, merge)
            frames = frames.permute(0, 2, 4, 3, 5, 1).reshape(gated.shape[0], self.hidden_size)
        else:
            outputs = []
            lengths = [height * width for height, width in frame_shapes]
            for frame, (height, width) in zip(gated.split(lengths), frame_shapes):
                frame = frame.view(height // merge, width // merge, merge, merge, self.hidden_size)
                frame = frame.permute(4, 0, 2, 1, 3).reshape(1, self.hidden_size, height, width)
                frame = self.spatial_conv(frame)
                frame = frame.view(self.hidden_size, height // merge, merge, width // merge, merge)
                outputs.append(frame.permute(1, 3, 2, 4, 0).reshape(height * width, self.hidden_size))
            frames = torch.cat(outputs)
        return self.out_proj(scale * frames)


class Lfm3VisionLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention:
            self.self_attn = Lfm3VisionAttention(config)
        else:
            self.conv = Lfm3VisionConv2d(config)
        self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size)
        self.operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x, lengths, frame_shapes, cos, sin):
        residual = x
        hidden = self.operator_norm(x)
        hidden = self.self_attn(hidden, lengths, cos, sin) if self.is_attention else self.conv(hidden, frame_shapes)
        hidden = hidden + residual
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class Lfm3VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patchifier = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.rotary = RoPE2D(vision_inv_freq(config))
        self.layers = nn.ModuleList([Lfm3VisionLayer(config, index) for index in range(config.num_hidden_layers)])

    def forward(self, pixel_values, grid_thw):
        frame_shapes = vision_frame_shapes(grid_thw)
        lengths = [height * width for height, width in frame_shapes]
        hidden = self.embedding_norm(self.patchifier(pixel_values))
        cos, sin = self.rotary(vision_position_ids(grid_thw, self.spatial_merge_size))
        for layer in self.layers:
            hidden = layer(hidden, lengths, frame_shapes, cos, sin)
        return hidden


class Lfm3VisionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample_factor = config.downsample_factor
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * config.downsample_factor**2,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = get_activation(getattr(config, "projector_hidden_act", "gelu"))
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=config.projector_bias)

    def forward(self, image_features):
        hidden = self.layer_norm(image_features).reshape(-1, self.linear_1.in_features)
        return self.linear_2(self.act(self.linear_1(hidden)))


class Lfm3VlVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patchifier = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.rotary_pos_emb = RoPE2D(vision_inv_freq(config))
        self.layers = nn.ModuleList([Lfm3VisionLayer(config, layer_index) for layer_index in range(config.num_hidden_layers)])

    def forward(self, pixel_values, grid_thw):
        positions = vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = vision_cu_seqlens(grid_thw)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        frame_shapes = vision_frame_shapes(grid_thw)
        cos, sin = self.rotary_pos_emb(positions)

        hidden_states = self.embedding_norm(self.patchifier(pixel_values.to(self.patchifier.weight.dtype)))
        for layer in self.layers:
            hidden_states = layer(hidden_states, lengths, frame_shapes, cos, sin)
        return hidden_states


class Lfm3VlMultiModalProjector(Lfm3VisionProjector):
    pass
