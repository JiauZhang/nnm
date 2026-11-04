import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, apply_rope, rotary_inv_freq
from nnm.models.pretrained import PretrainedModel

LAYER_KINDS = {
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
    "variate_time": (False, True),
}

ROLE_TARGET = 0
ROLE_PAST_COVARIATE = 1
ROLE_KNOWN_COVARIATE = 2


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.output = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.residual = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        return self.output(F.silu(self.hidden(x))) + self.residual(x)


class SwiGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias):
        super().__init__()
        self.gate_up = nn.Linear(embed_dim, 2 * hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(gate * F.silu(up))


def scaled_dot_product_attention(query, key, value, mask, is_causal=False):
    batch, seq_len, num_heads, head_dim = query.shape
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    return out.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)


class TwoStreamMask:
    def __init__(self, mask, enc_len):
        self.mask = mask
        self.enc_len = enc_len


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, is_time, bias, gate):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_time = is_time
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=bias) if gate == "sdpa_sigmoid" else None

    def forward(self, x, mask, cos, sin):
        rows, patches, _ = x.shape
        hidden = x if self.is_time else x.transpose(0, 1)
        qkv = self.qkv_proj(hidden).reshape(*hidden.shape[:2], 3, self.head_dim, self.num_heads)
        query, key, value = qkv.permute(2, 0, 1, 4, 3).unbind(0)
        if self.is_time:
            query, key = apply_rope(query, cos, sin, dim=2), apply_rope(key, cos, sin, dim=2)
        if isinstance(mask, TwoStreamMask):
            split = mask.enc_len
            enc = scaled_dot_product_attention(
                query[:, :split], key[:, :split], value[:, :split], mask.mask[:, :, :split, :split]
            )
            dec = scaled_dot_product_attention(query[:, split:], key, value, mask.mask[:, :, split:])
            out = torch.cat([enc, dec], dim=1)
        else:
            out = scaled_dot_product_attention(query, key, value, mask, is_causal=self.is_time and mask is None)
        if self.gate is not None:
            out = out * self.gate(hidden).sigmoid()
        out = (
            out.reshape(rows, patches, self.embed_dim)
            if self.is_time
            else out.reshape(patches, rows, self.embed_dim).transpose(0, 1)
        )
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, kind, embed_dim, num_heads, mlp_hidden_dim, norm_eps, attn_bias, mlp_bias, time_gate, variate_gate):
        super().__init__()
        axes = LAYER_KINDS[kind]
        self.attn_norms = nn.ModuleList([RMSNorm(embed_dim, norm_eps) for _ in axes])
        self.attns = nn.ModuleList(
            [Attention(embed_dim, num_heads, is_time, attn_bias, time_gate if is_time else variate_gate) for is_time in axes]
        )
        self.mlp_norm = RMSNorm(embed_dim, norm_eps)
        self.mlp = SwiGLU(embed_dim, mlp_hidden_dim, mlp_bias)

    def forward(self, x, time_mask, variate_mask, cos, sin):
        for norm, attn in zip(self.attn_norms, self.attns):
            x = x + attn(norm(x), time_mask if attn.is_time else variate_mask, cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    kind=kind,
                    embed_dim=config.d_model,
                    num_heads=config.num_heads,
                    mlp_hidden_dim=config.mlp_hidden_dim,
                    norm_eps=config.norm_eps,
                    attn_bias=config.attn_bias,
                    mlp_bias=config.mlp_bias,
                    time_gate=config.time_attn_gate,
                    variate_gate=config.variate_attn_gate,
                )
                for kind in config.block_schedule
            ]
        )
        head_dim = config.d_model // config.num_heads
        self.rope = RoPE(rotary_inv_freq(head_dim, config.rope_theta, int(head_dim * config.rope_partial_factor)))
        self.out_norm = RMSNorm(config.d_model, config.norm_eps)

    def forward(self, x, time_mask, variate_mask, positions):
        cos, sin = self.rope(positions)
        for layer in self.layers:
            x = layer(x, time_mask, variate_mask, cos, sin)
        return self.out_norm(x)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = ResidualBlock(2 * patch_size, 4 * d_model, d_model, bias=True)
        self.forecast_query = nn.Parameter(torch.zeros(d_model))

    def forward(self, scaled, observed):
        rows, length = scaled.shape
        patches = length // self.patch_size
        values = scaled.reshape(rows, patches, self.patch_size)
        missing = (~observed).reshape(rows, patches, self.patch_size).to(scaled.dtype)
        return self.patch_proj(torch.cat([values, missing], dim=-1))

    def forecast_tokens(self, rows, patches):
        return self.forecast_query.view(1, 1, -1).expand(rows, patches, -1)


def monotone_quantiles(raw, levels, width_factor):
    median_index = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(median_index + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(median_index):
        widths[i] = levels[i + 1] - levels[i]
    widths = torch.tensor(widths, device=raw.device, dtype=raw.dtype)
    increment = width_factor * widths * F.softplus(raw)
    median = raw[..., median_index : median_index + 1]
    upper = torch.cumsum(increment[..., median_index + 1 :], dim=-1)
    lower = torch.flip(torch.cumsum(torch.flip(increment[..., :median_index], dims=[-1]), dim=-1), dims=[-1])
    limit = torch.finfo(raw.dtype).max
    return torch.cat([median - lower, median, median + upper], dim=-1).clamp(min=-limit, max=limit)


class QuantileHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.levels = list(config.quantile_knots)
        self.width_factor = config.quantile_width_factor
        self.projection = ResidualBlock(config.d_model, 4 * config.d_model, config.patch_size * len(self.levels), bias=False)

    def forward(self, x):
        raw = self.projection(x)
        *leading, patches, _ = raw.shape
        raw = raw.reshape(*leading, patches, self.patch_size, len(self.levels))
        raw = raw.to(torch.promote_types(raw.dtype, torch.float32))
        raw = monotone_quantiles(raw, self.levels, self.width_factor)
        return raw.reshape(*leading, patches * self.patch_size, len(self.levels))


def causal_statistics(data, observed, minimum_scale):
    weights = torch.ones_like(data) * observed
    with torch.no_grad():
        x = data.to(torch.float64)
        w = weights.to(torch.float64)
        weighted = w * x
        cum_w = torch.cumsum(w, dim=-1)
        cum_x = torch.cumsum(weighted, dim=-1)
        denominator = cum_w.clamp_min(1.0)
        means = cum_x / denominator
        shifted = torch.zeros_like(means)
        shifted[..., 1:] = means[..., :-1]
        increment = (x - shifted) * (x - means) * w
        m2 = torch.cumsum(increment, dim=-1)
        variance = m2 / torch.clamp(denominator - 1.0, min=1.0)
        scales = torch.sqrt(variance).clamp(min=minimum_scale)
    return means.to(data.dtype), scales.to(data.dtype)


def causal_patch_loc_scale(data, observed, patch_size, minimum_scale):
    means, scales = causal_statistics(data, observed, minimum_scale)
    patch_means = means.unfold(-1, patch_size, patch_size)[..., -1]
    patch_scales = scales.unfold(-1, patch_size, patch_size)[..., -1]
    return patch_means.repeat_interleave(patch_size, dim=-1), patch_scales.repeat_interleave(patch_size, dim=-1)


def apply_affine_asinh(data, loc, scale, observed, use_asinh):
    out_dtype = torch.promote_types(torch.promote_types(data.dtype, loc.dtype), scale.dtype)
    work = torch.promote_types(out_dtype, torch.float32)
    scaled = (data.to(work) - loc.to(work)) / scale.to(work).clamp_min(torch.finfo(work).eps)
    scaled = scaled.masked_fill(~observed, 0.0)
    if use_asinh:
        scaled = torch.where(observed, scaled.asinh(), scaled)
    return scaled.to(out_dtype)


def invert_affine_asinh(x, loc, scale, use_asinh):
    out_dtype = torch.promote_types(torch.promote_types(x.dtype, loc.dtype), scale.dtype)
    work = torch.promote_types(out_dtype, torch.float32)
    x = x.to(work)
    if use_asinh:
        x = x.sinh()
    return (x * scale.to(work) + loc.to(work)).to(out_dtype)


def clamp_nonfinite(values):
    clamped = torch.where(
        values == float("inf"),
        torch.where(values.isfinite(), values, -float("inf")).amax(dim=-1, keepdim=True),
        torch.where(
            values == -float("inf"),
            torch.where(values.isfinite(), values, float("inf")).amin(dim=-1, keepdim=True),
            values,
        ),
    )
    return torch.where(clamped.isinf(), torch.zeros_like(clamped), clamped)


def denormalize_quantiles(predictions, loc, scale, use_asinh):
    return clamp_nonfinite(invert_affine_asinh(predictions, loc, scale, use_asinh))


def enc_key_validity(observed_per_patch, attend_missing):
    if not attend_missing:
        return observed_per_patch
    return observed_per_patch | observed_per_patch.cummax(dim=-1).values


def enc_causal_mask(key_valid):
    length = key_valid.shape[-1]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=key_valid.device))
    eye = torch.eye(length, dtype=torch.bool, device=key_valid.device)
    return (causal[None] & key_valid[:, None, :]) | eye[None]


def forecast_time_mask(context_patches, horizon, key_valid, device):
    rows = key_valid.shape[0]
    enc = enc_causal_mask(key_valid)
    enc_dec = torch.zeros(rows, context_patches, horizon, dtype=torch.bool, device=device)
    dec_enc = key_valid[:, None, :].expand(rows, horizon, context_patches)
    dec_dec = torch.tril(torch.ones(horizon, horizon, dtype=torch.bool, device=device))[None].expand(rows, horizon, horizon)
    return torch.cat([torch.cat([enc, enc_dec], dim=2), torch.cat([dec_enc, dec_dec], dim=2)], dim=1).unsqueeze(1)


def forecast_positions(context_patches, horizon, first_observed):
    positions = torch.arange(context_patches + horizon, device=first_observed.device)
    return positions[None, :] - first_observed[:, None]


def two_stream_time_mask(min_context, patches, block_size, key_valid, bidirectional_rows):
    device = key_valid.device
    position = torch.arange(patches, device=device)[None, :]
    offset = (position - min_context[:, None]).clamp(min=0)
    anchor = min_context[:, None] + block_size * (offset // block_size)
    active = position >= min_context[:, None]
    anchor = torch.where(active, anchor, torch.zeros_like(anchor))
    causal = torch.tril(torch.ones(patches, patches, dtype=torch.bool, device=device))
    key_column = key_valid[:, None, :]
    index = torch.arange(patches, device=device).view(1, 1, patches)
    eye = torch.eye(patches, dtype=torch.bool, device=device)
    enc_enc = enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        enc_bi = key_valid[:, None, :].expand(key_valid.shape[0], patches, patches) | eye[None]
        enc_enc = torch.where(bidirectional_rows[:, None, None], enc_bi, enc_enc)
    enc_dec = torch.zeros(key_valid.shape[0], patches, patches, dtype=torch.bool, device=device)
    dec_enc = active[:, :, None] & (index < anchor[:, :, None]) & key_column
    same_block = anchor[:, :, None] == anchor[:, None, :]
    dec_dec = same_block & causal[None] & active[:, :, None] & active[:, None, :] | eye[None]
    return torch.cat([torch.cat([enc_enc, enc_dec], dim=2), torch.cat([dec_enc, dec_dec], dim=2)], dim=1).unsqueeze(1)


def variate_mask(group_ids, key_valid, horizon, target_rows=None, bidirectional_rows=None):
    rows = key_valid.shape[0]
    if rows == 1:
        return None
    device = key_valid.device
    same_group = group_ids[:, None] == group_ids[None, :]
    dec_valid = torch.ones(rows, horizon, dtype=torch.bool, device=device)
    slot_valid = torch.cat([key_valid, dec_valid], dim=1).permute(1, 0)
    mask = same_group[None, None] & slot_valid[:, None, None, :]
    forbid = torch.zeros(rows, rows, dtype=torch.bool, device=device)
    if target_rows is not None:
        forbid = forbid | ((~target_rows[:, None]) & target_rows[None, :])
    if bidirectional_rows is not None:
        forbid = forbid | (bidirectional_rows[:, None] & (~bidirectional_rows)[None, :])
    mask = mask & ~forbid[None, None]
    return mask | torch.eye(rows, dtype=torch.bool, device=device)[None, None]


def two_stream_positions(patches, first_observed):
    position = torch.arange(patches, device=first_observed.device)
    row = position[None, :] - first_observed[:, None]
    return torch.cat([row, row], dim=-1)


def first_observed_patch(observed_per_patch, group_ids):
    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    index = inverse[:, None].expand(-1, observed_per_patch.shape[-1])
    grouped = torch.zeros(unique_ids.numel(), observed_per_patch.shape[-1], device=observed_per_patch.device)
    grouped.scatter_reduce_(0, index, observed_per_patch.float(), reduce="amax", include_self=True)
    return grouped.argmax(dim=-1)[inverse]


class Timestral(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.representation = PatchEmbedding(config.patch_size, config.d_model)
        self.transformer = Transformer(config)
        self.head = QuantileHead(config)

    def forward(self, inputs, observed, group_ids, horizon_patches, role_ids=None, known_future=None, known_future_mask=None):
        if role_ids is None or not self.config.covariate_enabled or bool((role_ids == ROLE_TARGET).all()):
            return self._predict_context_only(inputs, observed, group_ids, horizon_patches)
        return self._predict_window(inputs, observed, group_ids, horizon_patches, role_ids, known_future, known_future_mask)

    def _scale(self, inputs, observed):
        loc, scale = causal_patch_loc_scale(
            inputs.unsqueeze(1), observed.unsqueeze(1), self.config.patch_size, self.config.minimum_scale
        )
        loc, scale = loc.squeeze(1), scale.squeeze(1)
        scaled = apply_affine_asinh(inputs, loc, scale, observed, self.config.use_asinh)
        return scaled, loc, scale

    def _predict_context_only(self, inputs, observed, group_ids, horizon_patches):
        patch_size = self.config.patch_size
        rows, length = inputs.shape
        context_patches, horizon = length // patch_size, horizon_patches
        scaled, loc, scale = self._scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, horizon)
        observed_per_patch = observed.unflatten(-1, (context_patches, patch_size)).any(dim=-1)
        key_valid = enc_key_validity(observed_per_patch, self.config.attend_missing_patches)
        time_mask = forecast_time_mask(context_patches, horizon, key_valid, inputs.device)
        v_mask = variate_mask(group_ids, key_valid, horizon)
        positions = forecast_positions(context_patches, horizon, first_observed_patch(observed_per_patch, group_ids))
        predictions = self._run(x_enc, x_dec, time_mask, v_mask, positions)
        anchor_loc = loc.unflatten(-1, (context_patches, patch_size))[..., 0][:, -1:]
        anchor_scale = scale.unflatten(-1, (context_patches, patch_size))[..., 0][:, -1:]
        return denormalize_quantiles(predictions, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), self.config.use_asinh)

    def _predict_window(self, inputs, observed, group_ids, horizon_patches, role_ids, known_future, known_future_mask):
        patch_size = self.config.patch_size
        rows, length = inputs.shape
        horizon_length = horizon_patches * patch_size
        if known_future is None:
            known_future = torch.zeros(rows, horizon_length, device=inputs.device, dtype=inputs.dtype)
            known_future_mask = torch.zeros(rows, horizon_length, device=inputs.device, dtype=torch.bool)
        full_inputs = torch.cat([inputs, known_future], dim=-1)
        full_observed = torch.cat([observed, known_future_mask], dim=-1)
        context_patches = length // patch_size
        patches = context_patches + horizon_patches
        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COVARIATE
        scaled, loc, scale = self._scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_covariate_scale(full_inputs, full_observed, scaled, loc, scale, known_rows)
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, patches)
        observed_per_patch = full_observed.unflatten(-1, (patches, patch_size)).any(dim=-1)
        key_valid = enc_key_validity(observed_per_patch, self.config.attend_missing_patches)
        min_context = torch.full((rows,), context_patches, device=inputs.device, dtype=torch.long)
        time_mask = two_stream_time_mask(min_context, patches, self.config.block_size, key_valid, known_rows)
        v_mask = variate_mask(group_ids, key_valid, patches, target_rows=target_rows, bidirectional_rows=known_rows)
        positions = two_stream_positions(patches, first_observed_patch(observed_per_patch, group_ids))
        predictions = self._run(x_enc, x_dec, time_mask, v_mask, positions).unflatten(1, (patches, patch_size))
        horizon_predictions = predictions[:, context_patches:].reshape(rows, horizon_length, predictions.shape[-1])
        anchor_loc = loc.unflatten(-1, (patches, patch_size))[..., 0][:, context_patches - 1 : context_patches]
        anchor_scale = scale.unflatten(-1, (patches, patch_size))[..., 0][:, context_patches - 1 : context_patches]
        return denormalize_quantiles(
            horizon_predictions, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), self.config.use_asinh
        )

    def _known_covariate_scale(self, inputs, observed, scaled, loc, scale, known_rows):
        if not bool(known_rows.any()):
            return scaled, loc, scale
        stat_dtype = torch.promote_types(inputs.dtype, torch.float32)
        x = inputs.to(stat_dtype)
        mask = observed.to(stat_dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        loc_c = (x * mask).sum(dim=-1, keepdim=True) / count
        variance = (((x - loc_c) * mask) ** 2).sum(dim=-1, keepdim=True) / count
        scale_c = variance.sqrt().clamp_min(self.config.minimum_scale)
        binary = ((inputs == 0.0) | (inputs == 1.0) | ~observed).all(dim=-1, keepdim=True) & observed.any(dim=-1, keepdim=True)
        loc_c = torch.where(binary, torch.zeros_like(loc_c), loc_c).to(inputs.dtype)
        scale_c = torch.where(binary, torch.ones_like(scale_c), scale_c).to(inputs.dtype)
        known_scaled = apply_affine_asinh(inputs, loc_c, scale_c, observed, use_asinh=False)
        if self.config.use_asinh:
            work = torch.promote_types(known_scaled.dtype, torch.float32)
            known_scaled = torch.where(observed & ~binary, known_scaled.to(work).asinh().to(known_scaled.dtype), known_scaled)
        selected = known_rows[:, None]
        return (
            torch.where(selected, known_scaled, scaled),
            torch.where(selected, loc_c, loc),
            torch.where(selected, scale_c, scale),
        )

    def _run(self, x_enc, x_dec, time_mask, v_mask, positions):
        x = torch.cat([x_enc, x_dec], dim=1)
        if self.config.split_context_horizon_attn:
            time_mask = TwoStreamMask(time_mask, x_enc.shape[1])
        return self.head(self.transformer(x, time_mask, v_mask, positions)[:, x_enc.shape[1] :])
