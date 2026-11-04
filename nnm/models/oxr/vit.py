import torch
from torch import nn
from torch.nn import functional as F

from nnm.layers.attention import ragged_attention
from nnm.layers.rope import RoPE2D, apply_rope
from nnm.layers.transformer import MLP


class NaViTPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.hidden_stride = config.hidden_stride
        self.num_patches = config.num_patches
        self.preserve_original_pe = config.preserve_original_pe

        if self.num_patches > 0:
            self.patch_embedding = nn.Linear(self.num_channels * self.patch_size**2, self.embed_dim)
            position_grid = int(self.num_patches**0.5)
        else:
            self.patch_embedding = nn.Conv2d(
                self.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
            )
            self.num_patches = (config.image_size // self.patch_size) ** 2
            position_grid = config.image_size // self.patch_size
        if self.preserve_original_pe:
            self.position_embedding_size = position_grid
            self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def _position_embedding(self, grid_thws, patch_embeds):
        size = self.position_embedding_size
        positional_embeddings = self.position_embedding.weight.reshape(size, size, -1)
        positional_embeddings = positional_embeddings.unsqueeze(0).permute(0, 3, 1, 2)
        output = torch.zeros_like(patch_embeds)

        offset = 0
        for grid_t, grid_h, grid_w in grid_thws:
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            pe = F.interpolate(positional_embeddings, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
            pe = pe.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1).repeat(grid_t, 1)
            pe = pe.reshape(
                grid_t,
                grid_h // self.hidden_stride,
                self.hidden_stride,
                grid_w // self.hidden_stride,
                self.hidden_stride,
                -1,
            )
            pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(grid_t * grid_h * grid_w, -1)
            count = grid_t * grid_h * grid_w
            output[offset : offset + count] = pe
            offset += count
        return output

    def forward(self, pixel_values, grid_thws=None):
        target_dtype = self.patch_embedding.weight.dtype
        if isinstance(self.patch_embedding, nn.Linear):
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        else:
            patches = pixel_values.to(dtype=target_dtype).view(
                -1, self.num_channels * self.temporal_patch_size, self.patch_size, self.patch_size
            )
            patch_embeds = self.patch_embedding(patches).reshape(-1, self.embed_dim)

        if self.preserve_original_pe:
            patch_embeds = patch_embeds + self._position_embedding(grid_thws, patch_embeds)
        return patch_embeds


class NaViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = config.use_rope
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None):
        seq_len = hidden_states.shape[0]
        query = self.q_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)

        if self.use_rope:
            cos, sin = position_embeddings
            query, key = apply_rope(query, cos, sin, dim=1), apply_rope(key, cos, sin, dim=1)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return self.out_proj(ragged_attention(query, key, value, lengths, scale=self.scale))


class NaViTEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = NaViTAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None):
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), cu_seqlens, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class NaViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_stride = config.hidden_stride
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.spatial_merge_unit = self.hidden_stride**2
        self.fullatt_block_indexes = self._parse_block_indexes(config.fullatt_block_indexes)
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        self.rotary_pos_emb = RoPE2D([inv_freq, inv_freq.clone()])
        self.layers = nn.ModuleList([NaViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    @staticmethod
    def _parse_block_indexes(indexes):
        if indexes is None:
            return None
        if isinstance(indexes, str):
            return [int(index) for index in indexes.split("|")]
        return list(indexes)

    def _rotary_pos_emb(self, grid_thws):
        pos_ids = []
        for grid_t, grid_h, grid_w in grid_thws:
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            hpos_ids = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
            hpos_ids = hpos_ids.reshape(
                grid_h // self.hidden_stride, self.hidden_stride, grid_w // self.hidden_stride, self.hidden_stride
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
            wpos_ids = wpos_ids.reshape(
                grid_h // self.hidden_stride, self.hidden_stride, grid_w // self.hidden_stride, self.hidden_stride
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(grid_t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def _window_index(self, grid_thws):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        merger_window_size = self.window_size // self.hidden_stride // self.patch_size

        for grid_t, grid_h, grid_w in grid_thws:
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            merged_h, merged_w = grid_h // self.hidden_stride, grid_w // self.hidden_stride
            index = torch.arange(grid_t * merged_h * merged_w).reshape(grid_t, merged_h, merged_w)
            pad_h = merger_window_size - merged_h % merger_window_size
            pad_w = merger_window_size - merged_w % merger_window_size
            windows_h = (merged_h + pad_h) // merger_window_size
            windows_w = (merged_w + pad_w) // merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(grid_t, windows_h, merger_window_size, windows_w, merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, windows_h * windows_w, merger_window_size, merger_window_size
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += grid_t * merged_h * merged_w

        return torch.cat(window_index, dim=0), cu_window_seqlens

    def forward(self, inputs_embeds, grid_thws, output_hidden_states=False):
        rotary_positions = self._rotary_pos_emb(grid_thws)
        window_index, cu_window_seqlens = self._window_index(grid_thws)
        cu_window_seqlens = torch.unique_consecutive(
            torch.tensor(cu_window_seqlens, device=inputs_embeds.device, dtype=torch.int32)
        )

        seq_len = inputs_embeds.shape[0]
        num_units = seq_len // self.spatial_merge_unit
        inputs_embeds = inputs_embeds.reshape(num_units, self.spatial_merge_unit, -1)[window_index].reshape(seq_len, -1)
        rotary_positions = rotary_positions.reshape(num_units, self.spatial_merge_unit, -1)[window_index].reshape(seq_len, -1)
        position_embeddings = self.rotary_pos_emb(rotary_positions)
        reverse_indices = torch.argsort(window_index)

        cu_seqlens = torch.repeat_interleave(grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]).cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens.to(torch.int32), (1, 0), value=0)

        hidden_states = inputs_embeds
        encoder_states = []
        for index, layer in enumerate(self.layers):
            use_window = self.fullatt_block_indexes is not None and index not in self.fullatt_block_indexes
            layer_seqlens = cu_window_seqlens if use_window else cu_seqlens
            hidden_states = layer(hidden_states, layer_seqlens, position_embeddings)
            if output_hidden_states:
                encoder_states.append(
                    hidden_states.reshape(num_units, self.spatial_merge_unit, -1)[reverse_indices].reshape(seq_len, -1)
                )

        hidden_states = hidden_states.reshape(num_units, self.spatial_merge_unit, -1)[reverse_indices].reshape(seq_len, -1)
        return hidden_states, tuple(encoder_states) if output_hidden_states else None


class Siglip2NaViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = NaViTPatchEmbedding(config)
        self.encoder = NaViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values, grid_thws, output_hidden_states=False):
        hidden_states = self.embeddings(pixel_values, grid_thws)
        last_hidden_state, encoder_states = self.encoder(hidden_states, grid_thws, output_hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state, encoder_states
