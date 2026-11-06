import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.attention import (
    CrossAttention,
    InterleavedRotarySelfAttention,
    SelfAttention,
    window_partition_padded,
    window_unpartition_padded,
)
from nnm.layers.conv import CXBlock
from nnm.layers.mlp import MLP
from nnm.layers.norm import LayerNorm2d
from nnm.layers.sinusoidal import sinusoidal_embedding
from nnm.layers.rope import axial_rope_cos_sin
from nnm.layers.transformer import CrossTransformerBlock, RotaryVisionBlock, TransformerBlock
from nnm.models.pretrained import PretrainedModel


def _mlp_head(input_size, hidden_size, output_size, num_layers=2):
    layers = []
    current = input_size
    for _ in range(num_layers - 1):
        layers += [nn.Linear(current, hidden_size), nn.ReLU()]
        current = hidden_size
    layers.append(nn.Linear(current, output_size))
    return nn.Sequential(*layers)


def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class Sam3SinePositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_1d_positions(self, x, y):
        embeddings = []
        for coordinate in (y, x):
            embedding = sinusoidal_embedding(coordinate * self.scale, self.num_pos_feats, self.temperature, interleaved=True)
            embeddings.append(embedding)
        return embeddings[1], embeddings[0]

    def encode_boxes(self, boxes):
        dim_t = self.temperature ** (
            2
            * torch.div(torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device), 2, rounding_mode="floor")
            / self.num_pos_feats
        )
        embeddings = []
        for index in range(4):
            angle = boxes[..., index, None] * self.scale / dim_t
            embeddings.append(torch.stack((angle[..., 0::2].sin(), angle[..., 1::2].cos()), dim=-1).flatten(-2))
        y, x, w, h = embeddings
        return torch.cat((y, x, w, h), dim=-1)

    def forward(self, shape, device, dtype, mask=None):
        height, width = shape[-2:]
        if mask is None:
            mask = torch.zeros((1, height, width), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        eps = 1e-6
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = self.temperature ** (
            2
            * torch.div(torch.arange(self.num_pos_feats, dtype=torch.int64, device=device), 2, rounding_mode="floor")
            / self.num_pos_feats
        )
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)


class Sam3VisionLayer(nn.Module):
    def __init__(self, config, window_size=0):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        image_size = config.get("image_size", 1008)
        patch_size = config.get("patch_size", 14)
        grid = image_size // patch_size
        norm_eps = config.get("layer_norm_eps", config.get("norm_eps", 1e-5))
        global_window = config.get("window_size", 24)
        dropout = config.get("hidden_dropout", 0.0)
        self.window_size = window_size
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.rope_scale = global_window / grid if window_size == 0 else 1.0
        self.block = RotaryVisionBlock(
            hidden_size,
            InterleavedRotarySelfAttention(
                hidden_size,
                num_heads,
                head_dim=hidden_size // num_heads,
                qkv_bias=True,
                o_bias=True,
                causal=False,
            ),
            MLP(
                hidden_size,
                config.get("intermediate_size", int(hidden_size * config.get("mlp_ratio", 4.625))),
                config.get("hidden_act", "gelu"),
                bias=True,
                dropout=dropout,
            ),
            norm_layer=partial(nn.LayerNorm, eps=norm_eps),
            layerscale_init=config.get("init_values", 0.0),
            drop_path_rate=config.get("drop_path_rate", 0.0),
            dropout=dropout,
        )

    def forward(self, hidden):
        height, width = hidden.shape[1], hidden.shape[2]
        padded_hw = None
        if self.window_size > 0:
            hidden, padded_hw = window_partition_padded(hidden, self.window_size)
            grid_height = grid_width = self.window_size
        else:
            grid_height, grid_width = height, width
        hidden = hidden.flatten(1, 2)
        cos, sin = axial_rope_cos_sin(
            self.block.attn.head_dim, self.rope_theta, grid_height, grid_width, self.rope_scale, device=hidden.device
        )
        hidden = self.block(hidden, rotary_embeddings=(cos, sin))
        hidden = hidden.reshape(-1, grid_height, grid_width, hidden.shape[-1])
        if self.window_size > 0:
            hidden = window_unpartition_padded(hidden, self.window_size, padded_hw, (height, width))
        return hidden


def sam3_get_abs_pos(abs_pos, has_cls_token, height, width, retain_cls_token, tiling=True):
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]
    size = int(abs_pos.shape[1] ** 0.5)
    if size != height or size != width:
        positions = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            positions = positions.tile([1, 1, height // size + 1, width // size + 1])[:, :, :height, :width]
        else:
            positions = F.interpolate(positions, size=(height, width), mode="bicubic", align_corners=False)
        if not retain_cls_token:
            return positions.permute(0, 2, 3, 1)
        return torch.cat([cls_pos, positions.permute(0, 2, 3, 1).reshape(1, height * width, -1)], dim=1)
    if not retain_cls_token:
        return abs_pos.reshape(1, height, width, -1)
    return torch.cat([cls_pos, abs_pos], dim=1)


class Sam3VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = config.get("patch_size", 14)
        self.patch_size = patch_size
        self.retain_cls_token = config.get("retain_cls_token", False)
        self.pretrain_use_cls_token = config.get("pretrain_use_cls_token", True)
        self.tile_abs_pos = config.get("tile_abs_pos", True)
        self.use_abs_pos = config.get("use_abs_pos", True)
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        pretrain_image_size = config.get("pretrain_image_size", 336)
        num_patches = (pretrain_image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            config.get("num_channels", config.get("in_chans", 3)),
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=config.get("bias_patch_embed", False),
        )
        if self.retain_cls_token:
            self.class_embedding = nn.Parameter(hidden_size**-0.5 * torch.randn(1, 1, hidden_size))
        if self.use_abs_pos:
            num_positions = num_patches + 1 if self.pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, hidden_size))
        self.dropout = nn.Dropout(config.get("hidden_dropout", 0.0))

    def forward(self, pixel_values):
        hidden = self.projection(pixel_values).permute(0, 2, 3, 1)
        height, width = hidden.shape[1], hidden.shape[2]
        if self.retain_cls_token:
            hidden = torch.cat([self.class_embedding, hidden.flatten(1, 2)], dim=1)
        if self.use_abs_pos:
            hidden = hidden + sam3_get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                height,
                width,
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )
        return hidden


class Sam3ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        window_size = config.get("window_size", 24)
        global_indexes = config.get("global_attn_indexes", [7, 15, 23, 31])
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        norm_eps = config.get("layer_norm_eps", config.get("norm_eps", 1e-5))
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        self.embeddings = Sam3VisionEmbeddings(config)
        self.retain_cls_token = self.embeddings.retain_cls_token
        self.ln_pre = nn.LayerNorm(hidden_size, eps=norm_eps) if config.get("ln_pre", True) else nn.Identity()
        self.ln_post = nn.LayerNorm(hidden_size, eps=norm_eps) if config.get("ln_post", False) else nn.Identity()
        self.layers = nn.ModuleList(
            [Sam3VisionLayer(config, 0 if index in global_indexes else window_size) for index in range(num_layers)]
        )

    def forward(self, pixel_values):
        hidden = self.ln_pre(self.embeddings(pixel_values))
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.ln_post(hidden)
        if self.retain_cls_token:
            hidden = hidden[:, 1:]
        return hidden


class Sam3FPNLayer(nn.Module):
    def __init__(self, in_channels, fpn_dim, scale_factor):
        super().__init__()
        layers = OrderedDict()
        if scale_factor == 4.0:
            layers["dconv_2x2_0"] = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            layers["gelu"] = nn.GELU()
            layers["dconv_2x2_1"] = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 2, stride=2)
            intermediate = in_channels // 4
        elif scale_factor == 2.0:
            layers["dconv_2x2"] = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            intermediate = in_channels // 2
        elif scale_factor == 1.0:
            intermediate = in_channels
        elif scale_factor == 0.5:
            layers["maxpool_2x2"] = nn.MaxPool2d(2, stride=2)
            intermediate = in_channels
        else:
            raise NotImplementedError(f"scale_factor={scale_factor} is not supported yet.")
        layers["conv_1x1"] = nn.Conv2d(intermediate, fpn_dim, 1)
        layers["conv_3x3"] = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.conv = nn.Sequential(layers)

    def forward(self, hidden):
        return self.conv(hidden)


class Sam3VisionNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.backbone_config.hidden_size
        fpn_dim = config.fpn_hidden_size
        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=fpn_dim // 2, normalize=True)
        self.fpn_layers = nn.ModuleList([Sam3FPNLayer(hidden_size, fpn_dim, scale) for scale in config.scale_factors])

    def forward(self, hidden):
        features = []
        positions = []
        for layer in self.fpn_layers:
            output = layer(hidden)
            features.append(output)
            positions.append(self.position_encoding(output.shape, output.device, output.dtype))
        return features, positions


class Sam3VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Sam3ViT(config.backbone_config)
        self.neck = Sam3VisionNeck(config)

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values)
        batch, height, width, _ = hidden.shape
        features, positions = self.neck(hidden.permute(0, 3, 1, 2))
        return {
            "last_hidden_state": hidden.view(batch, height * width, -1),
            "fpn_features": features,
            "fpn_positions": positions,
        }


class Sam3GeometryEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        norm_eps = config.get("layer_norm_eps", 1e-5)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.block = CrossTransformerBlock(
            hidden_size,
            SelfAttention(
                hidden_size,
                config.num_attention_heads,
                qkv_bias=True,
                o_bias=True,
                causal=False,
            ),
            CrossAttention(
                hidden_size,
                config.num_attention_heads,
                qkv_bias=True,
                o_bias=True,
            ),
            MLP(
                hidden_size,
                config.intermediate_size,
                config.get("hidden_act", "relu"),
                bias=True,
                dropout=config.get("dropout", 0.0),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, prompt, vision, vision_pos, prompt_mask=None):
        return self.block(prompt, vision, attn_mask=prompt_mask, memory_pos=vision_pos)


class Sam3GeometryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.roi_size = config.get("roi_size", 7)
        self.encode_boxes_as_points = config.get("encode_boxes_as_points", False)
        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        self.label_embed = nn.Embedding(6 if self.encode_boxes_as_points else 2, hidden_size)
        self.cls_embed = nn.Embedding(1, hidden_size)
        self.boxes_direct_project = nn.Linear(4, hidden_size)
        self.boxes_pool_project = nn.Conv2d(hidden_size, hidden_size, self.roi_size)
        self.boxes_pos_enc_project = nn.Linear(hidden_size + 2, hidden_size)
        self.points_direct_project = nn.Linear(2, hidden_size)
        self.points_pool_project = nn.Linear(hidden_size, hidden_size)
        self.points_pos_enc_project = nn.Linear(hidden_size, hidden_size)
        self.vision_layer_norm = nn.LayerNorm(hidden_size)
        self.img_pre_norm = nn.Identity()
        self.final_proj = nn.Linear(hidden_size, hidden_size)
        self.prompt_layer_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([Sam3GeometryEncoderLayer(config) for _ in range(config.num_layers)])
        self.output_layer_norm = nn.LayerNorm(hidden_size)

    def _encode_points(self, points, labels, vision):
        batch, num_points = points.shape[:2]
        embeddings = self.points_direct_project(points)
        grid = (points[:, :, None] * 2 - 1).to(vision.dtype)
        sampled = F.grid_sample(vision, grid, align_corners=False).squeeze(-1).permute(0, 2, 1)
        embeddings = embeddings + self.points_pool_project(sampled)
        x, y = points.unbind(-1)
        pos_x, pos_y = self.position_encoding.encode_1d_positions(x.flatten(), y.flatten())
        position = torch.cat([pos_x, pos_y], dim=1).view(batch, num_points, -1)
        embeddings = embeddings + self.points_pos_enc_project(position)
        return embeddings + self.label_embed(labels.long())

    def _encode_boxes(self, boxes, labels, vision):
        from torchvision.ops import roi_align  # noqa: PLC0415

        batch, num_boxes = boxes.shape[:2]
        height, width = vision.shape[-2:]
        embeddings = self.boxes_direct_project(boxes)
        xyxy = box_cxcywh_to_xyxy(boxes) * torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
        sampled = roi_align(vision, xyxy.unbind(0), self.roi_size)
        pooled = self.boxes_pool_project(sampled).view(batch, num_boxes, -1)
        embeddings = embeddings + pooled
        center_x, center_y, box_width, box_height = boxes.unbind(-1)
        pos_x, pos_y = self.position_encoding.encode_1d_positions(center_x.flatten(), center_y.flatten())
        position = torch.cat((pos_y, pos_x, box_height.flatten()[:, None], box_width.flatten()[:, None]), dim=1)
        position = position.view(batch, num_boxes, -1)
        embeddings = embeddings + self.boxes_pos_enc_project(position)
        return embeddings + self.label_embed(labels.long())

    def forward(self, boxes, labels, mask, features, positions, points=None, point_labels=None, point_mask=None):
        batch = features[-1].shape[0]
        vision = features[-1]
        vision_pos = positions[-1]
        normalized = self.vision_layer_norm(vision.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        prompt = self._encode_boxes(boxes, labels, normalized)
        prompt_mask = mask
        if points is not None:
            point_prompt = self._encode_points(points, point_labels, normalized)
            prompt = torch.cat([prompt, point_prompt], dim=1)
            prompt_mask = torch.cat([prompt_mask, point_mask], dim=1)
        cls = self.cls_embed.weight[:, None].expand(batch, -1, -1)
        prompt = torch.cat([prompt, cls], dim=1)
        prompt_mask = torch.cat([prompt_mask, torch.ones(batch, 1, dtype=prompt_mask.dtype, device=prompt_mask.device)], dim=1)
        prompt = self.prompt_layer_norm(self.final_proj(prompt))
        attn_mask = torch.where(prompt_mask[:, None, None, :], 0.0, float("-inf"))
        vision_flat = vision.flatten(2).transpose(1, 2)
        pos_flat = vision_pos.flatten(2).transpose(1, 2)
        for layer in self.layers:
            prompt = layer(prompt, vision_flat, pos_flat, attn_mask)
        return self.output_layer_norm(prompt), prompt_mask


class Sam3DetrEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        norm_eps = config.get("layer_norm_eps", 1e-5)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.block = CrossTransformerBlock(
            hidden_size,
            SelfAttention(
                hidden_size,
                config.num_attention_heads,
                qkv_bias=True,
                o_bias=True,
                causal=False,
            ),
            CrossAttention(
                hidden_size,
                config.num_attention_heads,
                qkv_bias=True,
                o_bias=True,
            ),
            MLP(
                hidden_size,
                config.intermediate_size,
                config.get("hidden_act", "relu"),
                bias=True,
                dropout=config.get("dropout", 0.0),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, vision, prompt, vision_pos, prompt_mask=None):
        return self.block(vision, prompt, cross_attn_mask=prompt_mask, self_pos=vision_pos)


class Sam3DetrEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Sam3DetrEncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, features, positions, prompt, prompt_mask=None):
        vision = features[-1].flatten(2).transpose(1, 2)
        position = positions[-1].flatten(2).transpose(1, 2)
        attn_mask = None
        if prompt_mask is not None:
            attn_mask = torch.where(prompt_mask[:, None, None, :], 0.0, float("-inf"))
        for layer in self.layers:
            vision = layer(vision, prompt, position, attn_mask)
        return {"last_hidden_state": vision, "pos_embeds": position, "text_features": prompt}


class Sam3DetrDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dropout = config.get("dropout", 0.0)
        norm_eps = config.get("layer_norm_eps", 1e-5)
        self.self_attn = SelfAttention(
            hidden_size,
            config.num_attention_heads,
            qkv_bias=True,
            o_bias=True,
            causal=False,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.text_cross_attn = CrossAttention(
            hidden_size,
            config.num_attention_heads,
            qkv_bias=True,
            o_bias=True,
        )
        self.text_cross_attn_dropout = nn.Dropout(dropout)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.vision_cross_attn = CrossAttention(
            hidden_size,
            config.num_attention_heads,
            qkv_bias=True,
            o_bias=True,
        )
        self.vision_cross_attn_dropout = nn.Dropout(dropout)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = MLP(
            hidden_size,
            config.intermediate_size,
            config.get("hidden_act", "relu"),
            bias=True,
            dropout=dropout,
        )
        self.mlp_layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, hidden, query_pos, text, vision, vision_pos, text_mask=None, vision_mask=None):
        query_pos = F.pad(query_pos, (0, 0, 1, 0), mode="constant", value=0)
        residual = hidden
        hidden = residual + self.self_attn_dropout(self.self_attn(hidden, qk_pos=(query_pos, None)))
        hidden = self.self_attn_layer_norm(hidden)
        residual = hidden
        hidden = residual + self.text_cross_attn_dropout(
            self.text_cross_attn(hidden, text, attn_mask=text_mask, qk_pos=(query_pos, None))
        )
        hidden = self.text_cross_attn_layer_norm(hidden)
        residual = hidden
        hidden = residual + self.vision_cross_attn_dropout(
            self.vision_cross_attn(hidden, vision, attn_mask=vision_mask, qk_pos=(query_pos, vision_pos))
        )
        hidden = self.vision_cross_attn_layer_norm(hidden)
        residual = hidden
        return self.mlp_layer_norm(residual + self.mlp_dropout(self.mlp(hidden)))


class Sam3DetrDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_queries = config.get("num_queries", 200)
        self.layers = nn.ModuleList([Sam3DetrDecoderLayer(config) for _ in range(config.num_layers)])
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.box_head = _mlp_head(hidden_size, hidden_size, 4, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_size)
        self.reference_points = nn.Embedding(self.num_queries, 4)
        self.presence_token = nn.Embedding(1, hidden_size)
        self.presence_head = _mlp_head(hidden_size, hidden_size, 1, 3)
        self.presence_layer_norm = nn.LayerNorm(hidden_size)
        self.presence_clamp = 10.0
        self.ref_point_head = _mlp_head(2 * hidden_size, hidden_size, hidden_size, 2)
        self.box_rpb_embed_x = _mlp_head(2, hidden_size, config.num_attention_heads, 2)
        self.box_rpb_embed_y = _mlp_head(2, hidden_size, config.num_attention_heads, 2)
        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=False)

    def _relative_position_bias(self, boxes, height, width):
        xyxy = box_cxcywh_to_xyxy(boxes)
        batch, num_queries, _ = xyxy.shape
        coords_h = torch.arange(height, dtype=boxes.dtype, device=boxes.device) / height
        coords_w = torch.arange(width, dtype=boxes.dtype, device=boxes.device) / width
        deltas_y = coords_h.view(1, -1, 1) - xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(batch, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(batch, num_queries, -1, 2)

        def log_scale(value):
            scaled = value * 8
            return torch.sign(scaled) * torch.log2(torch.abs(scaled) + 1.0) / math.log2(8)

        deltas_x = self.box_rpb_embed_x(log_scale(deltas_x))
        deltas_y = self.box_rpb_embed_y(log_scale(deltas_y))
        bias = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
        bias = bias.flatten(2, 3).permute(0, 3, 1, 2).contiguous()
        return bias

    def forward(self, vision, vision_pos, text, text_mask=None, spatial_shapes=None):
        batch = vision.shape[0]
        queries = self.query_embed.weight[None].expand(batch, -1, -1)
        boxes = self.reference_points.weight[None].expand(batch, -1, -1).sigmoid()
        presence = self.presence_token.weight[None].expand(batch, -1, -1)
        hidden = torch.cat([presence, queries], dim=1)
        text_attn_mask = None
        if text_mask is not None:
            text_attn_mask = torch.where(text_mask[:, None, None, :], 0.0, float("-inf"))
        hidden_outputs = []
        reference_outputs = [boxes]
        presence_outputs = []
        for layer in self.layers:
            sine = self.position_encoding.encode_boxes(boxes)
            query_pos = self.ref_point_head(sine)
            vision_mask = None
            if spatial_shapes is not None and spatial_shapes.shape[0] == 1:
                bias = self._relative_position_bias(boxes, spatial_shapes[0, 0], spatial_shapes[0, 1])
                vision_mask = F.pad(bias, (0, 0, 1, 0), mode="constant", value=0)
            hidden = layer(hidden, query_pos, text, vision, vision_pos, text_attn_mask, vision_mask)
            query_hidden = hidden[:, 1:]
            delta = self.box_head(self.output_layer_norm(query_hidden))
            boxes = (delta + inverse_sigmoid(boxes)).sigmoid()
            boxes = boxes.detach()
            hidden_outputs.append(self.output_layer_norm(query_hidden))
            reference_outputs.append(boxes)
            presence_outputs.append(
                self.presence_head(self.presence_layer_norm(hidden[:, :1]))
                .squeeze(-1)
                .clamp(-self.presence_clamp, self.presence_clamp)
            )
        return {
            "intermediate_hidden_states": torch.stack(hidden_outputs),
            "reference_boxes": torch.stack(reference_outputs[:-1]),
            "presence_logits": torch.stack(presence_outputs),
        }


class Sam3DotProductScoring(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.0):
        super().__init__()
        self.text_mlp = _mlp_head(hidden_size, intermediate_size, hidden_size, 2)
        self.text_mlp_dropout = nn.Dropout(dropout)
        self.text_mlp_out_norm = nn.LayerNorm(hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**-0.5
        self.clamp_max = 12.0

    def forward(self, hidden_states, text, text_mask=None):
        residual = text
        text = self.text_mlp_out_norm(residual + self.text_mlp_dropout(self.text_mlp(text)))
        if text_mask is None:
            pooled = text.mean(dim=1)
        else:
            valid = text_mask.to(text.dtype).unsqueeze(-1)
            pooled = (text * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        scores = torch.matmul(self.query_proj(hidden_states), self.text_proj(pooled).unsqueeze(-1))
        return (scores * self.scale).clamp(-self.clamp_max, self.clamp_max)


class Sam3PixelDecoder(nn.Module):
    def __init__(self, hidden_size, num_stages):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(hidden_size, hidden_size, 3, padding=1) for _ in range(num_stages)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden_size) for _ in range(num_stages)])

    def forward(self, features):
        hidden = features[-1]
        for index, feature in enumerate(reversed(features[:-1])):
            hidden = F.interpolate(hidden, size=feature.shape[-2:], mode="nearest") + feature
            hidden = F.relu(self.norms[index](self.conv_layers[index](hidden)))
        return hidden


class Sam3MaskDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dropout = config.get("dropout", 0.0)
        self.pixel_decoder = Sam3PixelDecoder(hidden_size, config.get("num_upsampling_stages", 3))
        self.mask_embedder = _mlp_head(hidden_size, hidden_size, hidden_size, num_layers=3)
        self.instance_projection = nn.Conv2d(hidden_size, hidden_size, 1)
        self.semantic_projection = nn.Conv2d(hidden_size, 1, 1)
        self.prompt_cross_attn = CrossAttention(hidden_size, config.num_attention_heads, qkv_bias=True, o_bias=True)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden_size)
        self.prompt_cross_attn_dropout = nn.Dropout(dropout)

    def forward(self, queries, features, encoder_hidden, prompt=None, prompt_mask=None):
        if prompt is not None:
            normed = self.prompt_cross_attn_norm(encoder_hidden)
            attn_mask = None
            if prompt_mask is not None:
                attn_mask = torch.where(prompt_mask[:, None, None, :], 0.0, float("-inf"))
            encoder_hidden = encoder_hidden + self.prompt_cross_attn_dropout(
                self.prompt_cross_attn(normed, prompt, attn_mask=attn_mask)
            )
        spatial = features[-1].shape[-2] * features[-1].shape[-1]
        batch, _, hidden_size = encoder_hidden.shape
        height, width = features[-1].shape[-2:]
        encoded = encoder_hidden[:, :spatial].transpose(1, 2).reshape(batch, hidden_size, height, width)
        pixel_features = list(features)
        pixel_features[-1] = encoded
        pixel = self.pixel_decoder(pixel_features)
        instance = self.instance_projection(pixel)
        embeddings = self.mask_embedder(queries)
        masks = torch.einsum("bqc,bchw->bqhw", embeddings, instance)
        return {"pred_masks": masks, "semantic_seg": self.semantic_projection(pixel)}


class SAM3(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        vision_config = config.get("vision_config")
        text_config = config.get("text_config")
        geometry_config = config.get("geometry_encoder_config")
        encoder_config = config.get("detr_encoder_config")
        decoder_config = config.get("detr_decoder_config")
        mask_config = config.get("mask_decoder_config")

        self.vision_encoder = Sam3VisionModel(vision_config)
        self.setup_text(config, text_config, encoder_config)
        self.geometry_encoder = Sam3GeometryEncoder(geometry_config)
        self.detr_encoder = Sam3DetrEncoder(encoder_config)
        self.detr_decoder = Sam3DetrDecoder(decoder_config)
        self.mask_decoder = Sam3MaskDecoder(mask_config)
        self.dot_product_scoring = Sam3DotProductScoring(
            decoder_config.hidden_size,
            decoder_config.get("intermediate_size", decoder_config.hidden_size * 4),
            decoder_config.get("dropout", 0.0),
        )

    def setup_text(self, config, text_config, encoder_config):
        hidden_size = text_config.get("hidden_size", text_config.get("d_model", 768))
        self.text_encoder = nn.Module()
        self.text_encoder.embeddings = nn.Module()
        vocab_size = text_config.get("vocab_size", 49408)
        max_positions = text_config.get("max_position_embeddings", 77)
        self.text_encoder.embeddings.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.text_encoder.embeddings.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.text_encoder.encoder = nn.Module()
        num_heads = text_config.get("num_attention_heads", 12)
        intermediate = text_config.get("intermediate_size", hidden_size * 4)
        norm_eps = text_config.get("layer_norm_eps", 1e-5)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.text_encoder.encoder.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    SelfAttention(
                        hidden_size,
                        num_heads,
                        qkv_bias=True,
                        o_bias=True,
                        causal=False,
                    ),
                    MLP(hidden_size, intermediate, text_config.get("hidden_act", "gelu"), bias=True),
                    norm_layer=norm_layer,
                )
                for _ in range(text_config.get("num_hidden_layers", 12))
            ]
        )
        self.text_encoder.final_layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.text_projection = nn.Linear(hidden_size, encoder_config.hidden_size)

    def get_text_features(self, input_ids, attention_mask=None):
        embeddings = self.text_encoder.embeddings
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = embeddings.token_embedding(input_ids) + embeddings.position_embedding(positions)
        mask = None
        if attention_mask is not None:
            mask = torch.where(attention_mask.bool()[:, None, None, :], 0.0, float("-inf"))
        for layer in self.text_encoder.encoder.layers:
            hidden = layer(hidden, attn_mask=mask)
        hidden = self.text_encoder.final_layer_norm(hidden)
        return {"last_hidden_state": hidden, "pooler_output": self.text_projection(hidden)}

    def get_vision_features(self, pixel_values):
        return self.vision_encoder(pixel_values)

    def forward(
        self,
        pixel_values,
        input_ids=None,
        attention_mask=None,
        text_embeds=None,
        input_boxes=None,
        input_boxes_labels=None,
    ):
        vision = self.vision_encoder(pixel_values)
        features = vision["fpn_features"][:-1]
        positions = vision["fpn_positions"][:-1]
        if text_embeds is None:
            text = self.get_text_features(input_ids, attention_mask)["pooler_output"]
        else:
            text = text_embeds
        text_mask = attention_mask.bool() if attention_mask is not None else None
        prompt = text
        prompt_mask = text_mask
        if input_boxes is not None and input_boxes.numel() > 0:
            labels = (
                input_boxes_labels
                if input_boxes_labels is not None
                else torch.ones(input_boxes.shape[:2], dtype=torch.long, device=input_boxes.device)
            )
            box_mask = (
                input_boxes_labels != -10
                if input_boxes_labels is not None
                else torch.ones(input_boxes.shape[:2], dtype=torch.bool, device=input_boxes.device)
            )
            labels = torch.where(labels == -10, torch.zeros_like(labels), labels)
            geometry, geometry_mask = self.geometry_encoder(input_boxes, labels, box_mask, features, positions)
            prompt = torch.cat([text, geometry], dim=1)
            if text_mask is None:
                text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            prompt_mask = torch.cat([text_mask, geometry_mask], dim=1)
        encoded = self.detr_encoder(features, positions, prompt, prompt_mask)
        decoded = self.detr_decoder(
            encoded["last_hidden_state"],
            encoded["pos_embeds"],
            encoded["text_features"],
            prompt_mask,
            torch.tensor([list(features[-1].shape[-2:])], device=pixel_values.device),
        )
        box_offsets = self.detr_decoder.box_head(decoded["intermediate_hidden_states"])
        boxes = (inverse_sigmoid(decoded["reference_boxes"]) + box_offsets).sigmoid()
        logits = self.dot_product_scoring(decoded["intermediate_hidden_states"], encoded["text_features"], prompt_mask).squeeze(
            -1
        )
        masks = self.mask_decoder(
            decoded["intermediate_hidden_states"][-1],
            features,
            encoded["last_hidden_state"],
            prompt,
            prompt_mask,
        )
        return {
            "pred_masks": masks["pred_masks"],
            "semantic_seg": masks["semantic_seg"],
            "pred_boxes": box_cxcywh_to_xyxy(boxes[-1]),
            "pred_logits": logits[-1],
            "presence_logits": decoded["presence_logits"][-1],
        }


class Sam3MaskDownSampler(nn.Module):
    def __init__(self, embed_dim=256, kernel_size=4, stride=4, total_stride=16, activation=nn.GELU):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        layers = []
        in_channels, out_channels = 1, 1
        for _ in range(num_layers):
            out_channels = out_channels * (stride**2)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            layers.append(LayerNorm2d(out_channels))
            layers.append(activation())
            in_channels = out_channels
        layers.append(nn.Conv2d(out_channels, embed_dim, 1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, masks):
        return self.encoder(masks)


class Sam3MaskEncoder(nn.Module):
    def __init__(self, hidden_size, out_dim=64, num_layers=2, norm_eps=1e-6):
        super().__init__()
        self.mask_downsampler = Sam3MaskDownSampler(hidden_size)
        self.pix_feat_proj = nn.Conv2d(hidden_size, hidden_size, 1)
        self.fuser = nn.ModuleList([CXBlock(hidden_size) for _ in range(num_layers)])
        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=out_dim // 2, normalize=True)
        self.out_proj = nn.Conv2d(hidden_size, out_dim, 1) if out_dim != hidden_size else nn.Identity()

    def forward(self, pix_feat, masks):
        masks = self.mask_downsampler(masks.sigmoid())
        masks = F.interpolate(masks, size=pix_feat.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.pix_feat_proj(pix_feat) + masks
        for layer in self.fuser:
            hidden = layer(hidden)
        hidden = self.out_proj(hidden)
        positions = self.position_encoding(hidden.shape, hidden.device, hidden.dtype)
        return hidden, positions


class Sam3MemoryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, intermediate_size=2048, norm_eps=1e-6):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.layers = nn.ModuleList(
            [
                CrossTransformerBlock(
                    hidden_size,
                    SelfAttention(
                        hidden_size,
                        num_heads,
                        qkv_bias=True,
                        o_bias=True,
                        causal=False,
                    ),
                    CrossAttention(hidden_size, num_heads, qkv_bias=True, o_bias=True),
                    MLP(hidden_size, intermediate_size, "gelu", bias=True),
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def forward(self, hidden, memory, memory_pos=None):
        for layer in self.layers:
            hidden = layer(hidden, memory, memory_pos=memory_pos)
        return self.norm(hidden)


class Sam3MultiplexMaskDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, multiplex_count, num_multimask_outputs=3, iou_hidden=256, norm_eps=1e-6):
        super().__init__()
        self.multiplex_count = multiplex_count
        self.num_mask_output_per_object = num_multimask_outputs + 1
        self.num_mask_tokens = multiplex_count * self.num_mask_output_per_object
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_size)
        self.iou_token = nn.Embedding(multiplex_count, hidden_size)
        self.obj_score_token = nn.Embedding(multiplex_count, hidden_size)
        self.transformer = Sam3MemoryAttention(hidden_size, num_heads, 2, intermediate_size=hidden_size * 4, norm_eps=norm_eps)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size // 4, 2, stride=2),
            LayerNorm2d(hidden_size // 4),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, 2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size // 8))
                for _ in range(self.num_mask_output_per_object)
            ]
        )
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, iou_hidden),
            nn.ReLU(),
            nn.Linear(iou_hidden, self.num_mask_output_per_object),
        )
        self.obj_score_head = nn.Linear(hidden_size, 1)

    def forward(self, image_embeddings, image_pe, per_object_embeddings=None):
        batch = image_embeddings.shape[0]
        image_tokens = image_embeddings.flatten(2).transpose(1, 2) + image_pe.flatten(2).transpose(1, 2)
        mask_tokens = self.mask_tokens.weight.view(1, self.multiplex_count, self.num_mask_output_per_object, -1).expand(
            batch, -1, -1, -1
        )
        if per_object_embeddings is not None:
            mask_tokens = mask_tokens + per_object_embeddings[:, :, None]
        iou_tokens = self.iou_token.weight[None].expand(batch, -1, -1)
        obj_tokens = self.obj_score_token.weight[None].expand(batch, -1, -1)
        queries = torch.cat([iou_tokens, obj_tokens, mask_tokens.flatten(1, 2)], dim=1)
        queries = self.transformer(queries, image_tokens)
        iou_out = queries[:, : self.multiplex_count]
        obj_out = queries[:, self.multiplex_count : 2 * self.multiplex_count]
        mask_out = queries[:, 2 * self.multiplex_count :].view(batch, self.multiplex_count, self.num_mask_output_per_object, -1)
        upscaled = self.output_upscaling(
            image_tokens.transpose(1, 2).reshape(batch, -1, image_embeddings.shape[-2], image_embeddings.shape[-1])
        )
        masks = torch.stack(
            [
                torch.einsum(
                    "bcl,bmc->bml",
                    upscaled.flatten(2),
                    self.output_hypernetworks_mlps[index](mask_out[:, :, index]),
                ).reshape(batch, self.multiplex_count, upscaled.shape[-2], upscaled.shape[-1])
                for index in range(self.num_mask_output_per_object)
            ],
            dim=2,
        )
        iou_pred = self.iou_prediction_head(iou_out)
        object_score_logits = self.obj_score_head(obj_out)
        return {"masks": masks, "iou_pred": iou_pred, "object_score_logits": object_score_logits}


class SAM31(SAM3):
    def __init__(self, config):
        super().__init__(config)
        decoder_config = config.get("detr_decoder_config")
        hidden_size = config.get("hidden_size", decoder_config.get("hidden_size", 256))
        self.memory_encoder = Sam3MaskEncoder(
            hidden_size,
            config.get("memory_dim", 64),
            config.get("memory_layers", 2),
        )
        self.memory_attention = Sam3MemoryAttention(
            hidden_size,
            config.get("num_attention_heads", config.get("num_heads", 8)),
            config.get("memory_attention_layers", 4),
            config.get("memory_intermediate_size", 2048),
        )
        self.multiplex_mask_decoder = Sam3MultiplexMaskDecoder(
            hidden_size,
            config.get("num_attention_heads", config.get("num_heads", 8)),
            config.get("multiplex_count", 8),
            config.get("num_multimask_outputs", 3),
            config.get("iou_hidden", 256),
        )

    def forward_video(self, pixel_values, masks, memory=None, memory_pos=None):
        vision = self.vision_encoder(pixel_values)
        features = vision["fpn_features"][:-1]
        positions = vision["fpn_positions"][:-1]
        memory_feature, memory_positions = self.memory_encoder(features[-1], masks)
        packed_memory = memory_feature.flatten(2).transpose(1, 2)
        packed_positions = memory_positions.flatten(2).transpose(1, 2)
        if memory is not None:
            packed_memory = torch.cat([memory, packed_memory], dim=1)
        hidden = self.memory_attention(packed_memory, packed_memory, packed_positions)
        decoded = self.multiplex_mask_decoder(features[-1], positions[-1])
        decoded["memory"] = hidden
        decoded["packed_positions"] = packed_positions
        return decoded
