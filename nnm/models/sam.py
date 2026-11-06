import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.activation import get_activation
from nnm.layers.attention import window_partition_padded, window_unpartition_padded
from nnm.layers.norm import LayerNorm2d
from nnm.layers.mlp import MLP
from nnm.layers.transformer import VisionBlock
from nnm.layers.vit import PatchEmbed2D, resize_abs_pos
from nnm.models.pretrained import PretrainedModel


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class SamPositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats, scale):
        super().__init__()
        self.scale = scale
        self.positional_embedding = nn.Parameter(scale * torch.randn(2, num_pos_feats))

    def forward(self, input_coords, input_shape=None):
        coordinates = input_coords.clone()
        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * math.pi * coordinates
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)

    def get_dense_pe(self, size):
        grid = torch.ones((size, size), device=self.positional_embedding.device, dtype=self.positional_embedding.dtype)
        y_embed = grid.cumsum(0) - 0.5
        x_embed = grid.cumsum(1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        positional_embedding = self(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)


class SamMaskEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", 256)
        mask_input_channels = config.get("mask_input_channels", 16)
        self.mask_input_channels = mask_input_channels // 4
        self.activation = get_activation(config.get("hidden_act", "gelu"))
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(mask_input_channels, hidden_size, kernel_size=1)
        eps = config.get("layer_norm_eps", 1e-6)
        self.layer_norm1 = LayerNorm2d(self.mask_input_channels, eps=eps)
        self.layer_norm2 = LayerNorm2d(mask_input_channels, eps=eps)

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.conv3(hidden_states)


class SamPromptEncoder(nn.Module):
    def __init__(self, vision_config, config):
        super().__init__()
        num_pos_feats = vision_config.get("num_pos_feats", 128)
        scale = vision_config.get("scale", vision_config.get("hidden_size", 768) // 2)
        self.shared_embedding = SamPositionalEmbedding(num_pos_feats, scale)
        self.mask_embed = SamMaskEmbedding(config)
        hidden_size = config.get("hidden_size", 256)
        self.hidden_size = hidden_size
        self.no_mask_embed = nn.Embedding(1, hidden_size)
        image_size = _pair(config.get("image_size", 1024))
        patch_size = _pair(config.get("patch_size", 16))
        self.image_embedding_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.input_image_size = image_size
        num_point_embeddings = config.get("num_point_embeddings", 4)
        self.point_embed = nn.ModuleList([nn.Embedding(1, hidden_size) for _ in range(num_point_embeddings)])
        self.not_a_point_embed = nn.Embedding(1, hidden_size)

    def embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device, dtype=points.dtype)
            padding_label = -torch.ones(target_labels_shape, device=labels.device, dtype=labels.dtype)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        point_embedding = self.shared_embedding(points, self.input_image_size)
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)
        point_embedding = torch.where(labels[..., None] != -10, point_embedding, torch.zeros_like(point_embedding))
        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )
        return point_embedding

    def embed_boxes(self, boxes):
        boxes = boxes + 0.5
        batch_size, num_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, num_boxes, 2, 2)
        corner_embedding = self.shared_embedding(coords, self.input_image_size)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def embed_masks(self, masks):
        return self.mask_embed(masks)

    def get_dense_pe(self):
        return self.shared_embedding.get_dense_pe(self.image_embedding_size[0])

    def forward(self, input_points=None, input_labels=None, input_boxes=None, input_masks=None):
        sparse_embeddings = None
        batch_size = 1
        if input_points is not None:
            batch_size = input_points.shape[0]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            sparse_embeddings = self.embed_points(input_points, input_labels, pad=(input_boxes is None))
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self.embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.embed_masks(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embeddings, dense_embeddings


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    rel_pos_resized = F.interpolate(
        rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
        size=max_rel_dist,
        mode="linear",
    )
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def get_decomposed_rel_pos(query, rel_pos_h, rel_pos_w, q_size, k_size):
    query_height, query_width = q_size
    key_height, key_width = k_size
    relative_position_height = get_rel_pos(query_height, key_height, rel_pos_h)
    relative_position_width = get_rel_pos(query_width, key_width, rel_pos_w)
    batch_size, _, dim = query.shape
    reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
    rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
    return rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]


class SamRelativePositionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias, use_rel_pos, input_size, attention_dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.dropout = attention_dropout
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, hidden_states, qk_pos=None, attn_mask=None, lengths=None):
        batch_size, height, width, _ = hidden_states.shape
        qkv = (
            self.qkv(hidden_states).reshape(batch_size, height * width, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)
        attn_weights = (query * self.scale) @ key.transpose(-2, -1)
        if self.use_rel_pos:
            rel_pos = get_decomposed_rel_pos(query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))
            attn_weights = attn_weights + rel_pos.reshape_as(attn_weights)
        attn_weights = torch.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_weights @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        return self.proj(attn_output)


class SamVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", 768)
        num_heads = config.get("num_attention_heads", 12)
        num_channels = config.get("num_channels", 3)
        image_size = _pair(config.get("image_size", 1024))
        patch_size = _pair(config.get("patch_size", 16))
        depth = config.get("num_hidden_layers", 12)
        mlp_ratio = config.get("mlp_ratio", 4.0)
        mlp_dim = config.get("mlp_dim")
        if mlp_dim is None:
            mlp_dim = int(hidden_size * mlp_ratio)
        norm_eps = config.get("layer_norm_eps", 1e-6)
        hidden_act = config.get("hidden_act", "gelu")
        qkv_bias = config.get("qkv_bias", True)
        use_abs_pos = config.get("use_abs_pos", True)
        use_rel_pos = config.get("use_rel_pos", True)
        window_size = config.get("window_size", 14)
        global_attn_indexes = config.get("global_attn_indexes", (2, 5, 8, 11))
        out_channels = config.get("output_channels", 256)
        attention_dropout = config.get("attention_dropout", 0.0)

        self.patch_embed = PatchEmbed2D(
            kernel_size=patch_size,
            in_channels=num_channels,
            embed_dim=hidden_size,
            bias=True,
            norm=False,
        )
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size[0], self.grid_size[1], hidden_size))
        self.window_sizes = [0 if index in global_attn_indexes else window_size for index in range(depth)]
        self.layers = nn.ModuleList(
            [
                VisionBlock(
                    hidden_size,
                    SamRelativePositionAttention(
                        hidden_size,
                        num_heads,
                        qkv_bias,
                        use_rel_pos,
                        self.grid_size if block_window == 0 else (block_window, block_window),
                        attention_dropout,
                    ),
                    MLP(hidden_size, mlp_dim, hidden_act, bias=True),
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                )
                for block_window in self.window_sizes
            ]
        )
        self.neck = nn.Module()
        self.neck.conv1 = nn.Conv2d(hidden_size, out_channels, kernel_size=1, bias=False)
        self.neck.layer_norm1 = LayerNorm2d(out_channels)
        self.neck.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.neck.layer_norm2 = LayerNorm2d(out_channels)

    def forward(self, pixel_values):
        hidden_states, (height, width) = self.patch_embed(pixel_values)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], height, width, -1)
        if self.pos_embed is not None:
            if self.pos_embed.shape[1] != height or self.pos_embed.shape[2] != width:
                pos_embed = resize_abs_pos(self.pos_embed, height)
            else:
                pos_embed = self.pos_embed
            hidden_states = hidden_states + pos_embed
        for layer, window in zip(self.layers, self.window_sizes):
            if window > 0:
                height, width = hidden_states.shape[1], hidden_states.shape[2]
                hidden_states, padding_shape = window_partition_padded(hidden_states, window)
                hidden_states = layer(hidden_states)
                hidden_states = window_unpartition_padded(hidden_states, window, padding_shape, (height, width))
            else:
                hidden_states = layer(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.neck.conv1(hidden_states)
        hidden_states = self.neck.layer_norm1(hidden_states)
        hidden_states = self.neck.conv2(hidden_states)
        return self.neck.layer_norm2(hidden_states)


class SamAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, downsample_rate=1):
        super().__init__()
        self.internal_dim = hidden_size // downsample_rate
        self.num_attention_heads = num_heads
        if self.internal_dim % num_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")
        self.scaling = (self.internal_dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, hidden_size)

    def separate_heads(self, hidden_states):
        batch, point_batch, num_tokens, channels = hidden_states.shape
        channels_per_head = channels // self.num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch, num_tokens, self.num_attention_heads, channels_per_head)
        return hidden_states.transpose(1, 2)

    def recombine_heads(self, hidden_states, point_batch_size):
        batch, num_tokens, num_heads, channels_per_head = hidden_states.shape
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, num_tokens, num_heads * channels_per_head)

    def forward(self, query, key, value, attention_similarity=None):
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        point_batch_size = query.shape[1]
        query = self.separate_heads(query)
        key = self.separate_heads(key)
        value = self.separate_heads(value)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        if attention_similarity is not None:
            attn_weights = attn_weights + attention_similarity
        attn_weights = torch.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_output = (attn_weights @ value).transpose(1, 2)
        attn_output = self.recombine_heads(attn_output, point_batch_size)
        return self.out_proj(attn_output)


class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate=2, skip_first_layer_pe=False):
        super().__init__()
        hidden_size = config.get("hidden_size", 256)
        num_heads = config.get("num_attention_heads", 8)
        mlp_dim = config.get("mlp_dim", 2048)
        eps = config.get("layer_norm_eps", 1e-6)
        hidden_act = config.get("hidden_act", "relu")
        self.hidden_size = hidden_size
        self.layer_norm_eps = eps
        self.self_attn = SamAttention(hidden_size, num_heads, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.cross_attn_token_to_image = SamAttention(hidden_size, num_heads, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = MLP(hidden_size, mlp_dim, hidden_act, bias=True)
        self.layer_norm3 = nn.LayerNorm(hidden_size, eps=eps)
        self.layer_norm4 = nn.LayerNorm(hidden_size, eps=eps)
        self.cross_attn_image_to_token = SamAttention(hidden_size, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_point_embedding, key_point_embedding, attention_similarity=None):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            query = queries + query_point_embedding
            queries = queries + self.self_attn(query, query, queries)
        queries = self.layer_norm1(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        queries = queries + self.cross_attn_token_to_image(query, key, keys, attention_similarity)
        queries = self.layer_norm2(queries)

        queries = queries + self.mlp(queries)
        queries = self.layer_norm3(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        keys = keys + self.cross_attn_image_to_token(key, query, queries)
        keys = self.layer_norm4(keys)
        return queries, keys


class SamTwoWayTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", 256)
        num_heads = config.get("num_attention_heads", 8)
        num_layers = config.get("num_hidden_layers", 2)
        attention_downsample_rate = config.get("attention_downsample_rate", 2)
        self.layers = nn.ModuleList(
            [
                SamTwoWayAttentionBlock(config, attention_downsample_rate, skip_first_layer_pe=(index == 0))
                for index in range(num_layers)
            ]
        )
        self.final_attn_token_to_image = SamAttention(hidden_size, num_heads, downsample_rate=attention_downsample_rate)
        self.layer_norm_final_attn = nn.LayerNorm(hidden_size)

    def forward(
        self, point_embeddings, image_embeddings, image_positional_embeddings, attention_similarity=None, target_embedding=None
    ):
        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        queries = point_embeddings
        keys = image_embeddings
        for layer in self.layers:
            if target_embedding is not None:
                queries = queries + target_embedding
            queries, keys = layer(
                queries,
                keys,
                point_embeddings,
                image_positional_embeddings,
                attention_similarity,
            )
        query = queries + point_embeddings
        key = keys + image_positional_embeddings
        queries = queries + self.final_attn_token_to_image(query, key, keys)
        queries = self.layer_norm_final_attn(queries)
        return queries, keys


class SamFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.activation(self.proj_in(hidden_states))
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))
        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = torch.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", 256)
        num_multimask_outputs = config.get("num_multimask_outputs", 3)
        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_size)
        self.transformer = SamTwoWayTransformer(config)
        self.upscale_conv1 = nn.ConvTranspose2d(hidden_size, hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(hidden_size // 4)
        self.activation = nn.GELU()
        self.output_hypernetworks_mlps = nn.ModuleList(
            [SamFeedForward(hidden_size, hidden_size, hidden_size // 8, 3) for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = SamFeedForward(
            hidden_size,
            config.get("iou_head_hidden_dim", 256),
            self.num_mask_tokens,
            config.get("iou_head_depth", 3),
        )

    def predict_masks(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        attention_similarity=None,
        target_embedding=None,
    ):
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1] if sparse_prompt_embeddings is not None else 1
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)
        if sparse_prompt_embeddings is not None:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_pe = image_pe.repeat_interleave(point_batch_size, 0)
        point_embeddings, image_embeddings = self.transformer(
            point_embeddings,
            image_embeddings,
            image_pe,
            attention_similarity,
            target_embedding,
        )
        iou_token_out = point_embeddings[:, :, 0, :]
        mask_tokens_out = point_embeddings[:, :, 1 : (1 + self.num_mask_tokens), :]

        image_embeddings = image_embeddings.transpose(2, 3).reshape(batch_size * point_batch_size, num_channels, height, width)
        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in = torch.stack(
            [mlp(mask_tokens_out[:, :, index, :]) for index, mlp in enumerate(self.output_hypernetworks_mlps)],
            dim=2,
        )
        _, upscaled_channels, upscaled_height, upscaled_width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(
            batch_size, point_batch_size, upscaled_channels, upscaled_height * upscaled_width
        )
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, upscaled_height, upscaled_width)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output=True,
        attention_similarity=None,
        target_embedding=None,
    ):
        masks, iou_pred = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            attention_similarity,
            target_embedding,
        )
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, :, mask_slice], iou_pred[:, :, mask_slice]


class SAM(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        vision_config = config.get("vision_config")
        prompt_encoder_config = config.get("prompt_encoder_config")
        mask_decoder_config = config.get("mask_decoder_config")
        num_pos_feats = vision_config.get("num_pos_feats", 128)
        scale = vision_config.get("scale", vision_config.get("hidden_size", 768) // 2)
        self.shared_image_embedding = SamPositionalEmbedding(num_pos_feats, scale)
        self.vision_encoder = SamVisionEncoder(vision_config)
        self.prompt_encoder = SamPromptEncoder(vision_config, prompt_encoder_config)
        self.shared_image_embedding.positional_embedding = self.prompt_encoder.shared_embedding.positional_embedding
        self.mask_decoder = SamMaskDecoder(mask_decoder_config)

    def get_dense_pe(self):
        return self.prompt_encoder.get_dense_pe()

    def forward(
        self,
        pixel_values,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        input_masks=None,
        multimask_output=True,
        attention_similarity=None,
        target_embedding=None,
    ):
        image_embeddings = self.vision_encoder(pixel_values)
        batch_size = image_embeddings.shape[0]
        image_pe = self.get_dense_pe().to(image_embeddings.dtype).repeat(batch_size, 1, 1, 1)
        if input_points is not None and input_labels is None:
            input_labels = torch.ones(input_points.shape[:3], dtype=torch.int, device=input_points.device)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(input_points, input_labels, input_boxes, input_masks)
        pred_masks, iou_scores = self.mask_decoder(
            image_embeddings,
            image_pe,
            sparse_embeddings,
            dense_embeddings,
            multimask_output,
            attention_similarity,
            target_embedding,
        )
        return {"pred_masks": pred_masks, "iou_scores": iou_scores, "image_embeddings": image_embeddings}
