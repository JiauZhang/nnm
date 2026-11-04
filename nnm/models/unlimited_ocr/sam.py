import torch
from torch import nn
from torch.nn import functional as F

from nnm.layers.vit import resize_abs_pos


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    def __init__(self, *, kernel_size, stride, in_chans, embed_dim, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class SamMlp(nn.Module):
    def __init__(self, *, embed_dim, mlp_dim):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


def get_abs_pos(abs_pos, target_size):
    source_size = abs_pos.shape[1]
    if source_size == target_size:
        return abs_pos
    return resize_abs_pos(abs_pos, target_size)


def window_partition(x, window_size):
    batch, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(batch, padded_h // window_size, window_size, padded_w // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows, (padded_h, padded_w)


def window_unpartition(windows, window_size, padded_hw, hw):
    padded_h, padded_w = padded_hw
    height, width = hw
    batch = windows.shape[0] // (padded_h * padded_w // window_size // window_size)
    x = windows.view(batch, padded_h // window_size, padded_w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, padded_h, padded_w, -1)
    if padded_h > height or padded_w > width:
        x = x[:, :height, :width, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        resized = F.interpolate(
            rel_pos.to(torch.float32).reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    rel_h = get_rel_pos(q_h, k_h, rel_pos_h)
    rel_w = get_rel_pos(q_w, k_w, rel_pos_w)
    batch, _, dim = q.shape
    r_q = q.reshape(batch, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rel_h).unsqueeze(-1)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rel_w).unsqueeze(-2)
    rel_h = rel_h.reshape(batch, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(batch, q_h * q_w, 1, k_w)
    return rel_h, rel_w


class SamAttention(nn.Module):
    def __init__(self, *, dim, num_heads, qkv_bias=True, use_rel_pos=False, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    def forward(self, x):
        batch, height, width, _ = x.shape
        qkv = self.qkv(x).reshape(batch, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, batch * self.num_heads, height * width, -1).unbind(0)
        attn_bias = None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))
            q = q.view(batch, self.num_heads, height * width, -1)
            k = k.view(batch, self.num_heads, height * width, -1)
            v = v.view(batch, self.num_heads, height * width, -1)
            rel_h = rel_h.view(batch, self.num_heads, rel_h.shape[1], rel_h.shape[2], rel_h.shape[3])
            rel_w = rel_w.view(batch, self.num_heads, rel_w.shape[1], rel_w.shape[2], rel_w.shape[3])
            attn_bias = (rel_h + rel_w).view(batch, self.num_heads, rel_h.shape[2], rel_h.shape[3] * rel_w.shape[4])
        else:
            q = q.view(batch, self.num_heads, height * width, -1)
            k = k.view(batch, self.num_heads, height * width, -1)
            v = v.view(batch, self.num_heads, height * width, -1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x = x.view(batch, self.num_heads, height, width, -1).permute(0, 2, 3, 1, 4).reshape(batch, height, width, -1)
        return self.proj(x)


class SamBlock(nn.Module):
    def __init__(
        self, *, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_eps=1e-6, use_rel_pos=False, window_size=0, input_size
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SamAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.mlp = SamMlp(embed_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        padded_hw = None
        hw = None
        if self.window_size > 0:
            height, width = x.shape[1], x.shape[2]
            hw = (height, width)
            x, padded_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, padded_hw, hw)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SamEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        in_chans=3,
        width,
        depth,
        num_heads,
        mlp_ratio=4.0,
        out_chans=256,
        window_size=14,
        global_attn_indexes=(),
        downsample_channels=(512, 1024),
        qkv_bias=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=width,
        )
        grid = image_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid, grid, width))
        self.blocks = nn.ModuleList(
            [
                SamBlock(
                    dim=width,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_eps=norm_eps,
                    use_rel_pos=True,
                    window_size=0 if index in global_attn_indexes else window_size,
                    input_size=(grid, grid),
                )
                for index in range(depth)
            ]
        )
        self.neck = nn.Sequential(
            nn.Conv2d(width, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )
        self.net_2 = nn.Conv2d(out_chans, downsample_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(downsample_channels[0], downsample_channels[1], kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + get_abs_pos(self.pos_embed, x.shape[1])
        for block in self.blocks:
            x = block(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        x = self.net_2(x)
        return self.net_3(x)
