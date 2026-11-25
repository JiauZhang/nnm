import itertools
from collections import deque
from functools import partial

import torch
from torch import nn

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.layers.attention import SelfAttention
from nnm.layers.grid import PatchEmbed2D
from nnm.layers.mlp import MLP
from nnm.layers.rope import RopePositionEmbedding
from nnm.layers.transformer import TransformerBlock
from nnm.models.dino import build_dinov3_block
from nnm.models.dpt import DPTDecoder
from nnm.pretrained import PretrainedModel

INPUT_MULTIPLE = 32
DEFAULT_WINDOW = 16
DEFAULT_OVERLAP = 4
MAX_ASPECT = 1.7778


def autocast_for(device, precision):
    return partial(torch.autocast, device_type=device.type, dtype=precision, enabled=precision != torch.float32)


class CrossFrameCache:
    def __init__(self, window):
        self.window = window
        self.seen = {}
        self.prefix = {}
        self.kv = {}

    def update_prefix(self, idx, prefix):
        if idx not in self.prefix:
            self.seen[idx] = 0
            self.prefix[idx] = prefix.new_empty((self.window, *prefix.shape))
        seen = self.seen[idx]
        self.prefix[idx][seen % self.window] = prefix
        return self.prefix[idx][: min(seen + 1, self.window)].mean(dim=0)

    def update(self, idx, key, value, pooled_key, pooled_value):
        num_prefix, num_patches = pooled_key.shape[2], key.shape[2]
        length = num_prefix + self.window * num_patches
        if idx not in self.kv:
            self.kv[idx] = key.new_empty((2, 1, key.shape[1], length, key.shape[3]))
        kv, seen = self.kv[idx], self.seen[idx]
        slot = seen % self.window
        rows = slice(num_prefix + slot * num_patches, num_prefix + (slot + 1) * num_patches)
        kv[0, :, :, rows], kv[1, :, :, rows] = key, value
        kv[0, :, :, :num_prefix], kv[1, :, :, :num_prefix] = pooled_key, pooled_value
        self.seen[idx] = seen + 1
        live = kv[:, :, :, : num_prefix + min(seen + 1, self.window) * num_patches]
        return live[0], live[1]


class MultiViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        layers = config.num_hidden_layers
        head_dim = hidden // heads
        num_register = config.get("num_register_tokens", 0)
        layerscale_init = config.get("layerscale_init", 1e-5)
        self.patch_size = config.get("patch_size", 16)
        self.num_prefix_tokens = 1 + num_register
        self.reassembled_blocks = tuple(config.reassembled_blocks)
        self.cross_frame_blocks = frozenset(config.cross_frame_blocks)

        self.patch_embed = PatchEmbed2D(
            kernel_size=self.patch_size,
            in_channels=config.get("num_channels", 3),
            embed_dim=hidden,
            bias=True,
            norm=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.reg_token = nn.Parameter(torch.zeros(1, num_register, hidden)) if num_register else None
        self.rope_embed = RopePositionEmbedding(
            hidden,
            heads,
            base=config.get("rope_theta", 100.0),
            min_period=config.get("pos_embed_rope_min_period"),
            max_period=config.get("pos_embed_rope_max_period"),
            normalize_coords=config.get("pos_embed_rope_normalize_coords", "max"),
        )
        self.rope_embed.register_buffer("periods", self.rope_embed.periods, persistent=False)
        self.blocks = nn.ModuleList(
            build_dinov3_block(config, hidden, heads, head_dim, layerscale_init, config.get("drop_path_rate", 0.0))
            for _ in range(layers)
        )

        dims = config.dims_reassemble
        self.reassemble = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden, dims[0], kernel_size=1), nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4)
                ),
                nn.Sequential(
                    nn.Conv2d(hidden, dims[1], kernel_size=1), nn.ConvTranspose2d(dims[1], dims[1], kernel_size=2, stride=2)
                ),
                nn.Sequential(nn.Conv2d(hidden, dims[2], kernel_size=1)),
                nn.Sequential(
                    nn.Conv2d(hidden, dims[3], kernel_size=1), nn.Conv2d(dims[3], dims[3], kernel_size=3, stride=2, padding=1)
                ),
            ]
        )

    def embed(self, x):
        tokens, grid = self.patch_embed(x)
        num_frames = tokens.shape[0]
        prefix = [self.cls_token.expand(num_frames, -1, -1)]
        if self.reg_token is not None:
            prefix.append(self.reg_token.expand(num_frames, -1, -1))
        return torch.cat(prefix + [tokens], dim=1), grid

    def qkv_rope(self, block, x, rotary=None):
        attn = block.attn
        query, key, value = attn.project(block.norm1(x))
        query, key, value = attn.split_heads(query, key, value)
        if rotary is not None:
            query, key = attn.apply_rotary(query, key, rotary)
        return query, key, value

    def finish_block(self, block, x, attended):
        attn = block.attn
        batch, length = x.shape[0], x.shape[1]
        merged = attn.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))
        x = x + block.dropout(block.drop_path(block.ls1(merged)))
        return x + block.dropout(block.drop_path(block.ls2(block.mlp(block.norm2(x)))))

    def cross_frame_attention(self, block, tokens, rotary):
        num_frames, _, dim = tokens.shape
        npt = self.num_prefix_tokens
        prefix, patches = tokens[:, :npt], tokens[:, npt:]
        x = torch.cat([prefix.mean(dim=0, keepdim=True), patches.reshape(1, -1, dim)], dim=1)
        x = block(x, rotary_embeddings=rotary)
        return torch.cat([prefix, x[:, npt:].reshape(num_frames, -1, dim)], dim=1)

    def cached_cross_frame_attention(self, idx, tokens, patch_rotary, pooled_rotary, cache):
        block = self.blocks[idx]
        npt = self.num_prefix_tokens
        prefix, patches = tokens[:, :npt], tokens[:, npt:]
        query, key, value = self.qkv_rope(block, patches, patch_rotary)
        pooled = cache.update_prefix(idx, prefix)
        _, pooled_key, pooled_value = self.qkv_rope(block, pooled, pooled_rotary)
        keys, values = cache.update(idx, key, value, pooled_key, pooled_value)
        attended = scaled_dot_product_attention(query, keys, values)
        return torch.cat([prefix, self.finish_block(block, patches, attended)], dim=1)

    def encode(self, x, cache=None):
        tokens, grid = self.embed(x)
        cos, sin = self.rope_embed(*grid)
        npt = self.num_prefix_tokens
        ones = cos.new_ones(npt, cos.shape[-1])
        zeros = sin.new_zeros(npt, sin.shape[-1])
        frame_rotary = (torch.cat([ones, cos], dim=0), torch.cat([zeros, sin], dim=0))
        if cache is not None:
            patch_rotary = (cos, sin)
            pooled_rotary = (ones, zeros)
        else:
            num_frames = tokens.shape[0]
            patch_rotary = (
                torch.cat([ones, cos.repeat(num_frames, 1)], dim=0),
                torch.cat([zeros, sin.repeat(num_frames, 1)], dim=0),
            )
            pooled_rotary = patch_rotary
        features = []
        for idx, block in enumerate(self.blocks):
            if idx not in self.cross_frame_blocks:
                tokens = block(tokens, rotary_embeddings=frame_rotary)
            elif cache is None:
                tokens = self.cross_frame_attention(block, tokens, patch_rotary)
            else:
                tokens = self.cached_cross_frame_attention(idx, tokens, patch_rotary, pooled_rotary, cache)
            if idx in self.reassembled_blocks:
                features.append(tokens[:, self.num_prefix_tokens :])
        return features, grid

    def reassemble_tokens(self, features, grid):
        num_frames = features[0].shape[0]
        return [
            layer(feature.transpose(1, 2).reshape(num_frames, -1, *grid).contiguous())
            for layer, feature in zip(self.reassemble, features)
        ]


def create_inverse_depth_head(dim_in):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_in // 2, kernel_size=3, padding=1),
        nn.ConvTranspose2d(dim_in // 2, dim_in // 2, kernel_size=2, stride=2),
        nn.Conv2d(dim_in // 2, 32, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1),
        nn.ReLU(),
    )


def sincos_2d(gh, gw, dim, device):
    g = max(gh, gw)
    ys = (torch.arange(gh, device=device) + 0.5 - gh / 2) / g
    xs = (torch.arange(gw, device=device) + 0.5 - gw / 2) / g
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    freq = torch.pi * 2.0 ** torch.linspace(0.0, 7.0, dim // 4, device=device)
    return torch.cat([f(c.reshape(-1, 1) * freq) for c in (yy, xx) for f in (torch.sin, torch.cos)], dim=-1)


class FOVHead(nn.Module):
    def __init__(self, embed_dim=384, num_taps=4, num_prefix=5, dim=248, depth=2, heads=4, pool_stride=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix = num_prefix
        self.dim = dim
        self.pool_stride = pool_stride
        self.tap_norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(num_taps))
        self.proj = nn.Linear(embed_dim * num_taps, dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.prefix_embed = nn.Parameter(torch.zeros(1, num_prefix, dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim,
                SelfAttention(dim, heads, qkv_bias=True, o_bias=True, causal=False),
                MLP(dim, dim * 4, "gelu", True),
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 2)

    def forward(self, tokens, grid):
        num_frames = tokens.shape[0]
        gh, gw = grid
        taps = tokens.float().split(self.embed_dim, dim=-1)
        x = self.proj(torch.cat([norm(t) for norm, t in zip(self.tap_norms, taps)], dim=-1))
        prefix, x = x[:, : self.num_prefix], x[:, self.num_prefix :]
        x = x.transpose(1, 2).reshape(num_frames, self.dim, gh, gw)
        x = torch.nn.functional.avg_pool2d(x, self.pool_stride, ceil_mode=True)
        x = x.flatten(2).transpose(1, 2) + sincos_2d(*x.shape[-2:], self.dim, x.device)
        query = self.query.expand(num_frames, -1, -1)
        x = torch.cat([query, prefix + self.prefix_embed, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, 0]))


def crop_angles_to_full(angles, crop_hw, full_hw):
    (ch, cw), (h, w) = crop_hw, full_hw
    a = torch.deg2rad(angles.to(torch.promote_types(angles.dtype, torch.float32)))
    fx = (cw / 2) / torch.tan(a[..., 0] / 2)
    fy = (ch / 2) / torch.tan(a[..., 1] / 2)
    out = torch.stack([2 * torch.atan((w / 2) / fx), 2 * torch.atan((h / 2) / fy)], dim=-1)
    return torch.rad2deg(out).to(angles.dtype)


def median_focal_length_px(fov_deg, height, width):
    hfov_deg, vfov_deg = fov_deg.median(dim=0).values
    if width < height:
        return 0.5 * width / torch.tan(torch.deg2rad(hfov_deg) / 2.0)
    return 0.5 * height / torch.tan(torch.deg2rad(vfov_deg) / 2.0)


class FOVEstimator(PretrainedModel):
    max_short_side = 512
    min_fov_deg = 1.0
    max_fov_deg = 179.0
    tapped_blocks = frozenset((2, 5, 8, 11))
    cross_frame_blocks = frozenset((5, 7, 9, 11))

    def __init__(self, config, precision=torch.float32):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        layers = config.num_hidden_layers
        head_dim = hidden // heads
        num_register = config.get("num_register_tokens", 0)
        layerscale_init = config.get("layerscale_init", 1e-5)
        trunk = nn.Module()
        trunk.patch_embed = PatchEmbed2D(
            kernel_size=config.get("patch_size", 16),
            in_channels=config.get("num_channels", 3),
            embed_dim=hidden,
            bias=True,
            norm=False,
        )
        trunk.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        trunk.reg_token = nn.Parameter(torch.zeros(1, num_register, hidden)) if num_register else None
        trunk.rope = RopePositionEmbedding(
            hidden,
            heads,
            base=config.get("rope_theta", 100.0),
            normalize_coords=config.get("pos_embed_rope_normalize_coords", "max"),
        )
        trunk.rope.register_buffer("periods", trunk.rope.periods, persistent=False)
        trunk.blocks = nn.ModuleList(
            build_dinov3_block(config, hidden, heads, head_dim, layerscale_init, config.get("drop_path_rate", 0.0))
            for _ in range(layers)
        )
        self.trunk = trunk
        self.num_prefix_tokens = 1 + num_register
        self.head = FOVHead(embed_dim=hidden, num_prefix=self.num_prefix_tokens)
        self.precision = precision

    def input_size(self, height, width):
        short = max(32, min(self.max_short_side, min(height, width) // 32 * 32))
        scale = short / min(height, width)
        return max(32, round(height * scale / 32) * 32), max(32, round(width * scale / 32) * 32)

    def resize(self, x):
        new_h, new_w = self.input_size(*x.shape[-2:])
        x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True)
        lo, hi = min(new_h, new_w), max(new_h, new_w)
        if hi <= MAX_ASPECT * lo:
            return x
        keep = max(32, int(MAX_ASPECT * lo) // 32 * 32)
        off = (hi - keep) // 2
        return x[..., off : off + keep] if new_w > new_h else x[..., off : off + keep, :]

    def encode(self, x):
        trunk = self.trunk
        tokens, grid = trunk.patch_embed(x)
        prefix = [trunk.cls_token.expand(tokens.shape[0], -1, -1)]
        if trunk.reg_token is not None:
            prefix.append(trunk.reg_token.expand(tokens.shape[0], -1, -1))
        tokens = torch.cat(prefix + [tokens], dim=1)
        cos, sin = trunk.rope(*grid)
        cos = torch.cat([cos.new_ones(self.num_prefix_tokens, cos.shape[-1]), cos], dim=0)
        sin = torch.cat([sin.new_zeros(self.num_prefix_tokens, sin.shape[-1]), sin], dim=0)
        npt = self.num_prefix_tokens
        taps = []
        for idx, block in enumerate(trunk.blocks):
            out = block(tokens, rotary_embeddings=(cos, sin))
            if idx in self.cross_frame_blocks:
                out = torch.cat([tokens[:, :npt], out[:, npt:]], dim=1)
            tokens = out
            if idx in self.tapped_blocks:
                taps.append(tokens)
        return torch.cat(taps, dim=-1), grid

    def forward(self, x):
        height, width = x.shape[-2:]
        y = self.resize(x)
        with autocast_for(x.device, self.precision)():
            tokens, grid = self.encode(y)
        logits = self.head(tokens, grid)
        fov_deg = self.min_fov_deg + torch.sigmoid(logits) * (self.max_fov_deg - self.min_fov_deg)
        full_h, full_w = self.input_size(height, width)
        crop_h, crop_w = y.shape[-2:]
        if (crop_h, crop_w) == (full_h, full_w):
            return fov_deg
        crop_in_source = (crop_h * height / full_h, crop_w * width / full_w)
        return crop_angles_to_full(fov_deg, crop_in_source, (height, width))

    def focal_length_px(self, x):
        return median_focal_length_px(self.forward(x), *x.shape[-2:])


class DepthPro2(PretrainedModel):
    def __init__(self, config, precision=torch.float32):
        super().__init__()
        self.encoder = MultiViewEncoder(config)
        self.decoder = DPTDecoder(config.dims_reassemble, config.dim_decoder)
        self.head = create_inverse_depth_head(self.decoder.dim_out)
        self.fov = FOVEstimator(config, precision) if config.get("fov", False) else None
        self.precision = precision

    def forward(self, x, cache=None):
        with autocast_for(x.device, self.precision)():
            features, grid = self.encoder.encode(x, cache)
        inverse_depth = []
        for frame in range(x.shape[0]):
            tokens = [f[frame : frame + 1] for f in features]
            inverse_depth.append(self.decode_frame(tokens, grid))
        return torch.cat(inverse_depth)

    def decode_frame(self, features, grid):
        device = features[0].device
        with autocast_for(device, self.precision)():
            decoded = self.decoder(self.encoder.reassemble_tokens(features, grid))
        return self.head(decoded.float())

    def predict_inverse_depth(self, x, cache=None):
        height, width = x.shape[-2:]
        pad_h = -height % INPUT_MULTIPLE
        pad_w = -width % INPUT_MULTIPLE
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return self.forward(x, cache)[:, 0, :height, :width]

    @torch.no_grad()
    def infer(self, x, f_px=None):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"infer takes a tensor from transform(image), got {type(x).__name__}.")
        if x.ndim not in (3, 4) or x.shape[-3] != 3 or not x.is_floating_point():
            raise ValueError(f"infer takes a float (3, H, W) or (T, 3, H, W) tensor, got {x.dtype} {tuple(x.shape)}.")
        is_image = x.ndim == 3
        if is_image:
            x = x.unsqueeze(0)
        width = x.shape[-1]
        inverse_depth = self.predict_inverse_depth(x)
        if f_px is not None:
            f_px = torch.as_tensor(f_px, dtype=torch.float32, device=x.device).squeeze()
        elif self.fov is not None:
            f_px = self.fov.focal_length_px(x)
        depth = None if f_px is None else f_px / (width * inverse_depth)
        if is_image:
            inverse_depth = inverse_depth[0]
            depth = None if depth is None else depth[0]
        return {"inverse_depth": inverse_depth, "depth": depth, "focallength_px": f_px}

    def stream(self, window=DEFAULT_WINDOW, overlap=DEFAULT_OVERLAP, f_px=None, mode="buffered"):
        return DepthStream(self, window, overlap, f_px, mode=mode)

    @torch.no_grad()
    def infer_video(self, frames, window=DEFAULT_WINDOW, overlap=DEFAULT_OVERLAP, f_px=None, mode="buffered"):
        if f_px is None and self.fov is not None:
            frames = iter(frames)
            head = list(itertools.islice(frames, window))
            if not head:
                return
            f_px = self.fov.focal_length_px(torch.stack(head))
            frames = itertools.chain(head, frames)
        stream = self.stream(window, overlap, f_px, mode=mode)
        for frame in frames:
            yield from stream.step(frame)
        yield from stream.flush()


def blend_ramp(length, like):
    return torch.linspace(1, 0, length + 2, device=like.device, dtype=like.dtype)[1:-1].view(-1, 1, 1)


class DepthStream:
    def __init__(self, model, window, overlap, f_px, *, mode):
        if mode not in ("buffered", "unbuffered"):
            raise ValueError(f"mode must be 'buffered' or 'unbuffered', got {mode!r}.")
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}.")
        if mode == "buffered" and not 0 <= overlap < window:
            raise ValueError(f"overlap must be from 0 to window - 1 = {window - 1}, got {overlap}.")
        if mode == "unbuffered" and model.training:
            raise RuntimeError("Unbuffered mode is inference only; call .eval() on the model.")
        self.model = model
        self.window = window
        self._f_px = None if f_px is None else torch.as_tensor(f_px, dtype=torch.float32).squeeze()
        self._fovs = []
        self._cache = CrossFrameCache(window) if mode == "unbuffered" else None
        self._clip = window if self._cache is None else 1
        self._overlap = overlap if self._cache is None else 0
        self._frames = deque(maxlen=self._clip)
        self._held = []
        self._new = 0

    @torch.no_grad()
    def step(self, frame):
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"step takes one frame (3, H, W), got {tuple(frame.shape)}.")
        if self._frames and frame.shape != self._frames[-1].shape:
            raise ValueError(f"step takes frames of one size, got {tuple(frame.shape)} after {tuple(self._frames[-1].shape)}.")
        self._frames.append(frame)
        self._new += 1
        if len(self._held) + self._new < self._clip:
            return []
        return self._predict(len(self._frames), keep=self._overlap)

    @torch.no_grad()
    def flush(self):
        if not self._new:
            return self._emit(len(self._held), self._f_px)
        return self._predict(len(self._frames) if self._overlap else self._new, keep=0)

    def _predict(self, length, keep):
        frames = list(self._frames)[-length:]
        inverse_depth = self.model.predict_inverse_depth(torch.stack(frames), self._cache)
        f_px = self._focal_length(frames)
        inverse_depth = inverse_depth[length - len(self._held) - self._new :]
        held, self._new = len(self._held), 0
        if held:
            previous = torch.stack(self._held)
            ramp = blend_ramp(held, previous)
            self._held = list(ramp * previous + (1 - ramp) * inverse_depth[:held])
        self._held += list(inverse_depth[held:])
        return self._emit(len(self._held) - keep, f_px)

    def _focal_length(self, frames):
        if self._f_px is not None or self.model.fov is None:
            return self._f_px
        self._fovs.append(self.model.fov(torch.stack(frames)))
        fovs = torch.cat(self._fovs)
        f_px = median_focal_length_px(fovs, *frames[0].shape[-2:])
        if len(fovs) >= self.window:
            self._f_px = f_px
        return f_px

    def _emit(self, count, f_px):
        done, self._held = self._held[:count], self._held[count:]
        if f_px is not None and done:
            f_px = f_px.to(done[0].device)
        return [
            {
                "inverse_depth": inverse_depth,
                "depth": None if f_px is None else f_px / (inverse_depth.shape[-1] * inverse_depth),
                "focallength_px": f_px,
            }
            for inverse_depth in done
        ]
