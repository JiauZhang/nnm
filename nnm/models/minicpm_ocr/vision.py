import torch
import torch.nn.functional as F
from torch import nn

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.layers.transformer import MLP, VisionBlock


def vision_cu_seqlens(target_sizes):
    counts = target_sizes[:, 0] * target_sizes[:, 1]
    return F.pad(torch.cumsum(counts, dim=0, dtype=torch.int32), (1, 0))


def nearest_position_ids(target_sizes, num_patches_per_side):
    device = target_sizes.device
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device=device)
    positions = []
    for height, width in target_sizes.tolist():
        rows = torch.bucketize(torch.arange(height, device=device) / height, boundaries, right=True)
        cols = torch.bucketize(torch.arange(width, device=device) / width, boundaries, right=True)
        positions.append((rows[:, None] * num_patches_per_side + cols).flatten())
    return torch.cat(positions)


def vision_window_index(target_sizes, window_side):
    grid = F.pad(target_sizes, (1, 0), value=1)
    windows = []
    cu_seqlens = [0]
    offset = 0
    for _, height, width in grid.tolist():
        height, width = int(height), int(width)
        index = torch.arange(height * width, device=target_sizes.device).view(height, width)
        pad_h = window_side - height % window_side
        pad_w = window_side - width % window_side
        padded = F.pad(index, (0, pad_w, 0, pad_h), value=-1)
        padded = padded.view((height + pad_h) // window_side, window_side, (width + pad_w) // window_side, window_side)
        padded = padded.permute(0, 2, 1, 3).reshape(-1, window_side * window_side)
        counts = (padded != -1).sum(-1)
        flat = padded.flatten()
        windows.append(flat[flat != -1] + offset)
        cu_seqlens.extend((counts.cumsum(0) + cu_seqlens[-1]).tolist())
        offset += height * width
    window_index = torch.cat(windows)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=target_sizes.device)
    return window_index, torch.unique_consecutive(cu_seqlens)


class VisionAttention(nn.Module):
    def __init__(self, *, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, cu_seqlens=None):
        batch, seq_len = x.shape[:-1]
        shape = (batch, seq_len, self.num_attention_heads, self.head_dim)
        query = self.q_proj(x).view(shape).transpose(1, 2)
        key = self.k_proj(x).view(shape).transpose(1, 2)
        value = self.v_proj(x).view(shape).transpose(1, 2)

        if cu_seqlens is None:
            output = scaled_dot_product_attention(query, key, value, is_causal=False, scale=self.scaling)
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [torch.split(tensor, lengths, dim=2) for tensor in (query, key, value)]
            output = torch.cat(
                [scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scaling) for q, k, v in zip(*splits)],
                dim=2,
            )

        output = output.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        return self.out_proj(output)


class VisionLayer(VisionBlock):
    def __init__(self, *, hidden_size, intermediate_size, num_attention_heads, layer_norm_eps):
        super().__init__(
            hidden_size,
            layer_norm_eps,
            VisionAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads),
            MLP(hidden_size, intermediate_size, "gelu_pytorch_tanh", bias=True),
            norm_factory=lambda dim: nn.LayerNorm(dim, eps=layer_norm_eps),
        )


class VisionWindowMerger(nn.Module):
    def __init__(self, *, hidden_size, intermediate_size, num_attention_heads, layer_norm_eps, window_kernel_size):
        super().__init__()
        self.window_kernel_size = tuple(window_kernel_size)
        self.hidden_size = hidden_size
        self.window_hidden_size = hidden_size * self.window_kernel_size[0] * self.window_kernel_size[1]
        self.window_intermediate_size = intermediate_size * self.window_kernel_size[0] * self.window_kernel_size[1]
        self.attn = VisionAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        self.norm_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm = nn.LayerNorm(self.window_hidden_size, eps=layer_norm_eps)
        self.linear_1 = nn.Linear(self.window_hidden_size, self.window_intermediate_size)
        self.linear_2 = nn.Linear(self.window_intermediate_size, hidden_size)

    def forward(self, x, target_sizes):
        window_h, window_w = self.window_kernel_size
        window_size = window_h * window_w
        window_index, cu_seqlens = vision_window_index(target_sizes, window_h)

        normed = self.norm_1(x)[:, window_index]
        normed = self.attn(normed, cu_seqlens=cu_seqlens)
        merged = x[:, window_index] + normed

        patches = merged.view(-1, window_size, self.hidden_size)
        flat = self.norm(patches.flatten(1))
        flat = self.linear_2(F.gelu(self.linear_1(flat), approximate="tanh"))
        return (flat + patches.mean(dim=1)).unsqueeze(0)


class VisionEmbeddings(nn.Module):
    def __init__(self, *, hidden_size, num_channels, image_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.patch_embedding = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Embedding(self.num_patches_per_side**2, hidden_size)

    def forward(self, pixel_values, target_sizes):
        embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        positions = nearest_position_ids(target_sizes, self.num_patches_per_side)
        return embeddings + self.position_embedding(positions).unsqueeze(0)


class MiniCPMOcrVisionModel(nn.Module):
    def __init__(self, config, *, insert_layer_id):
        super().__init__()
        self.insert_layer_id = insert_layer_id
        hidden_size = config.hidden_size
        self.embeddings = VisionEmbeddings(
            hidden_size=hidden_size,
            num_channels=config.num_channels,
            image_size=config.image_size,
            patch_size=config.patch_size,
        )
        self.layers = nn.ModuleList(
            [
                VisionLayer(
                    hidden_size=hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.vit_merger = VisionWindowMerger(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            window_kernel_size=getattr(config, "window_kernel_size", (2, 2)),
        )

    def forward(self, pixel_values, target_sizes, use_vit_merger=True):
        hidden = self.embeddings(pixel_values, target_sizes)
        cu_seqlens = vision_cu_seqlens(target_sizes)

        for layer_index, layer in enumerate(self.layers):
            hidden = layer(hidden, cu_seqlens=cu_seqlens)
            if use_vit_merger and layer_index == self.insert_layer_id:
                hidden = self.vit_merger(hidden, target_sizes)
                target_sizes = target_sizes // 2
                cu_seqlens = vision_cu_seqlens(target_sizes)

        return self.norm(hidden)


class DownsampleMLP(nn.Module):
    def __init__(self, *, hidden_size, output_size):
        super().__init__()
        merged_size = hidden_size * 4
        self.norm = nn.LayerNorm(merged_size, eps=1e-6)
        self.linear_1 = nn.Linear(merged_size, merged_size)
        self.linear_2 = nn.Linear(merged_size, output_size)

    def forward(self, x):
        x = self.linear_1(self.norm(x))
        return self.linear_2(F.gelu(x))


class MiniCPMOcrMerger(nn.Module):
    def __init__(self, *, vision_hidden_size, text_hidden_size, merge_kernel_size, merger_times):
        super().__init__()
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.merger_times = merger_times
        blocks = [
            DownsampleMLP(hidden_size=vision_hidden_size, output_size=vision_hidden_size) for _ in range(merger_times - 1)
        ]
        blocks.append(DownsampleMLP(hidden_size=vision_hidden_size, output_size=text_hidden_size))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, target_sizes):
        merge_h, merge_w = self.merge_kernel_size
        start = 0
        outputs = []
        for height, width in target_sizes.tolist():
            num_patches = height * width
            merged_h, merged_w = height // merge_h, width // merge_w
            hidden = hidden_states[0, start : start + num_patches]
            hidden = hidden.view(merged_h, merge_h, merged_w, merge_w, -1)
            hidden = hidden.permute(0, 2, 1, 3, 4).reshape(merged_h * merged_w, -1)
            hidden = self.blocks[0](hidden)

            for block in self.blocks[1:]:
                height, width = height // merge_h, width // merge_w
                merged_h, merged_w = height // merge_h, width // merge_w
                hidden = hidden.view(merged_h, merge_h, merged_w, merge_w, -1)
                hidden = hidden.permute(0, 2, 1, 3, 4).reshape(merged_h * merged_w, -1)
                hidden = block(hidden)

            outputs.append(hidden)
            start += num_patches

        return outputs
