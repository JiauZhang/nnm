import re, torch
import torch.nn as nn
import torch.nn.functional as F
from nnm.layers.activation import get_activation
from nnm.backbone.pp_lcnet_v3 import LCNetV3Encoder, make_divisible
from conippets.config import Config


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.is_causal = False
        self.attention_dropout = config.attention_dropout
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.projection(attn_output)

        return attn_output, attn_weights


class MLP(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = get_activation(config.hidden_act)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.drop(hidden_state)
        hidden_state = self.fc2(hidden_state)
        hidden_state = self.drop(hidden_state)
        return hidden_state


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config=config, in_features=self.embed_dim, hidden_features=int(self.embed_dim * config.mlp_ratio))
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, activation="silu"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, input):
        hidden_state = self.conv(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class EncoderWithSVTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = make_divisible(config.backbone_config.block_configs[-1][-1][2] * config.backbone_config.scale, config.backbone_config.divisor)
        hidden_size = config.hidden_size
        conv_configs = [
            (in_channels, in_channels // 8, config.conv_kernel_size),
            (in_channels // 8, hidden_size, (1, 1)),
            (hidden_size, in_channels, (1, 1)),
            (2 * in_channels, in_channels // 8, config.conv_kernel_size),
            (in_channels // 8, hidden_size, (1, 1)),
        ]
        self.convs = nn.ModuleList([ConvLayer(in_ch, out_ch, kernel_size=k) for in_ch, out_ch, k in conv_configs])
        self.svtr_block = nn.ModuleList()
        for _ in range(config.depth):
            self.svtr_block.append(Block(config=config))
        self.norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = self.convs[0](hidden_states)
        hidden_states = self.convs[1](hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            hidden_states = block(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = self.convs[2](hidden_states)
        hidden_states = self.convs[3](torch.cat((residual, hidden_states), dim=1))
        hidden_states = self.convs[4](hidden_states)
        hidden_states = hidden_states.squeeze(2).transpose(1, 2)

        return hidden_states


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = LCNetV3Encoder(config.backbone_config)

    def forward(self, pixel_values):
        feature_maps = self.backbone(pixel_values)
        hidden_state = feature_maps[-1]
        hidden_state = F.avg_pool2d(hidden_state, (3, 2))
        return hidden_state


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = EncoderWithSVTR(config)
        self.head = nn.Linear(config.hidden_size, config.head_out_channels)

    def forward(self, hidden_states):
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.head(hidden_states)
        hidden_states = F.softmax(hidden_states, dim=2, dtype=torch.float32).to(hidden_states.dtype)
        return hidden_states


class PPOCRv5MobileRecognitionModel(nn.Module):
    hf_config = Config(
        backbone_config=dict(
            stem_channels=16,
            stem_stride=2,
            scale=0.95,
            out_features=["stage2", "stage3", "stage4", "stage5"],
            out_indices=[2, 3, 4, 5],
            divisor=16,
            block_configs=[
                [[3, 16, 32, 1, False]],
                [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
                [[3, 64, 128, [2, 1], False], [3, 128, 128, 1, False]],
                [
                    [3, 128, 256, [1, 2], False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                ],
                [[5, 256, 512, [2, 1], True], [5, 512, 512, 1, True], [5, 512, 512, [2, 1], False], [5, 512, 512, 1, False]],
            ],
            conv_symmetric_num=4,
            reduction=4,
            hidden_act="hardswish",
        ),
        hidden_act="silu",
        hidden_size=120,
        mlp_ratio=2.0,
        depth=2,
        head_out_channels=18385,
        conv_kernel_size=[1, 3],
        qkv_bias=True,
        num_attention_heads=8,
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
    )

    def __init__(self, config=hf_config):
        super().__init__()
        self.model = Model(config)
        self.head = Head(config)

    def forward(self, pixel_values):
        hidden_states = self.model(pixel_values)
        logits = self.head(hidden_states)
        return logits

    def load_hf_state_dict(self, hf_state_dict):
        REPLACEMENTS = [
            (r"^model\.backbone\.encoder\.", "model.backbone."),
            (r"\.convolution\.convolution(\.|$)", r".conv.conv\1"),
            (r"\.depthwise_convolution\.", ".dw_conv."),
            (r"\.pointwise_convolution\.", ".pw_conv."),
            (r"\.convolution(\.|$)", r".conv\1"),
            (r"\.conv_small_symmetric(\.|$)", ".convs.4."),
            (r"\.conv_symmetric\.(\d+)(\.|$)", r".convs.\1."),
            (r"\.act\.lab\.(\w+)$", r".act_\1"),
            (r"\.lab\.(\w+)$", r".\1"),
            (r"\.squeeze_excitation_module\.", ".se."),
            (r"\.se\.convolutions\.", ".se.convs."),
            (r"\.identity\.", ".bn."),
            (r"head\.encoder\.conv_block\.", "head.encoder.convs."),
        ]
        converted = {}
        for key, value in hf_state_dict.items():
            new_key = key
            for pattern, repl in REPLACEMENTS:
                new_key = re.sub(pattern, repl, new_key)
            converted[new_key] = value
        self.load_state_dict(converted)
