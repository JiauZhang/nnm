import torch
import torch.nn.functional as F
from functools import partial
from torch import nn

from nnm.layers.attention import RotarySelfAttention
from nnm.layers.mlp import GatedMLP
from nnm.layers.norm import RMSNorm, ZCRMSNorm
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.models.minicpm_ocr.cache import build_layers_cache


class RMSNormGated(RMSNorm):
    def forward(self, x, gate):
        return (super().forward(x) * F.silu(gate.float())).to(x.dtype)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, *, chunk_size=64, initial_state=None, output_final_state=False):
    dtype = query.dtype
    batch, seq_len, _, key_head_dim = key.shape
    num_value_heads, value_head_dim = value.shape[-2:]
    state_shape = (batch, num_value_heads, key_head_dim, value_head_dim)

    query, key, value, beta, g = [
        tensor.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for tensor in (query, key, value, beta, g)
    ]
    query = l2norm(query)
    key = l2norm(key)
    query = query * query.shape[-1] ** -0.5

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query, key, value = (F.pad(tensor, (0, 0, 0, pad)) for tensor in (query, key, value))
    beta, g = (F.pad(tensor, (0, pad)) for tensor in (beta, g))

    num_chunks = (seq_len + pad) // chunk_size
    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, key_beta, value_beta = [
        tensor.reshape(batch, -1, num_chunks, chunk_size, tensor.shape[-1]) for tensor in (query, key, key_beta, value_beta)
    ]
    g = g.reshape(batch, -1, num_chunks, chunk_size)

    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cumulative = g.cumsum(dim=3)
    pairwise = (cumulative.unsqueeze(4) - cumulative.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut_system = (key_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_key_beta = key_beta * cumulative.exp().unsqueeze(-1)

    new_values = torch.linalg.solve_triangular(ut_system, value_beta, upper=False, unitriangular=True)
    key_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_key_beta, upper=False, unitriangular=True)

    state = torch.zeros(state_shape, dtype=torch.float32, device=query.device) if initial_state is None else initial_state
    output = torch.zeros_like(new_values)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp().unsqueeze(-1).unsqueeze(-1)

    for index in range(num_chunks):
        updated_values = new_values[:, :, index] - key_cumdecay[:, :, index] @ state
        output[:, :, index] = query[:, :, index] @ state + intra[:, :, index] @ updated_values
        state = state * chunk_decay[:, :, index] + key[:, :, index].transpose(-1, -2) @ updated_values

    output = output.reshape(batch, num_value_heads, seq_len + pad, value_head_dim)[:, :, :seq_len]
    output = output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)
    return output, (state if output_final_state else None)


def recurrent_gated_delta_rule(query, key, value, g, beta, *, initial_state=None, output_final_state=False):
    dtype = query.dtype
    batch, seq_len, _, key_head_dim = key.shape
    num_value_heads, value_head_dim = value.shape[-2:]

    query, key, value, beta, g = [
        tensor.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for tensor in (query, key, value, beta, g)
    ]
    query = l2norm(query)
    key = l2norm(key)
    query = query * query.shape[-1] ** -0.5

    if initial_state is None:
        state = torch.zeros((batch, num_value_heads, key_head_dim, value_head_dim), dtype=torch.float32, device=query.device)
    else:
        state = initial_state
    output = torch.zeros_like(value)

    for index in range(seq_len):
        state = state * g[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * key[:, :, index].unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, index] - memory) * beta[:, :, index].unsqueeze(-1)
        state = state + key[:, :, index].unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * query[:, :, index].unsqueeze(-1)).sum(dim=-2)

    output = output.transpose(1, 2).contiguous().to(dtype)
    return output, (state if output_final_state else None)


class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_value_heads = config.linear_num_value_heads
        self.num_key_heads = config.linear_num_key_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.key_head_dim * self.num_key_heads
        self.value_dim = self.value_head_dim * self.num_value_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.conv_kernel_size, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.norm = RMSNormGated(self.value_head_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_value_heads, bias=False)

    def forward(self, x, cache=None):
        batch, seq_len = x.shape[:2]
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).view(batch, seq_len, -1, self.value_head_dim)
        beta_input = self.in_proj_b(x)
        alpha_input = self.in_proj_a(x)

        if cache is None:
            combined = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
        else:
            combined = cache.update_conv(mixed_qkv)
        mixed_qkv = F.conv1d(combined, self.conv.weight, None, groups=self.conv_dim)[..., -seq_len:]
        mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        ratio = self.num_value_heads // self.num_key_heads
        if ratio > 1:
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        beta = beta_input.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(alpha_input.float() + self.dt_bias)
        initial_state = cache.recurrent_state if cache is not None else None

        if cache is not None and initial_state is not None and seq_len == 1:
            output, state = recurrent_gated_delta_rule(
                query, key, value, g, beta, initial_state=initial_state, output_final_state=True
            )
        else:
            output, state = chunk_gated_delta_rule(
                query, key, value, g, beta, initial_state=initial_state, output_final_state=cache is not None
            )
        if cache is not None:
            cache.update_recurrent(state)

        output = output.reshape(-1, self.value_head_dim)
        output = self.norm(output, z.reshape(-1, self.value_head_dim))
        return self.out_proj(output.reshape(batch, seq_len, -1))


class TextAttention(RotarySelfAttention):
    def __init__(self, config):
        super().__init__(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            qkv_bias=config.attention_bias,
            o_bias=config.attention_bias,
            qk_norm=True,
            norm_layer=partial(ZCRMSNorm, eps=config.rms_norm_eps),
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            2 * config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

    def forward(self, x, rotary_embeddings=None, cache=None, attn_mask=None):
        batch, seq_len = x.shape[:2]
        query, key, value = self.project(x)
        query = query.view(batch, seq_len, self.num_heads, 2 * self.head_dim)
        query, gate = query.chunk(2, dim=-1)
        gate = gate.reshape(batch, seq_len, -1)
        query = query.reshape(batch, seq_len, -1)
        query, key, value = self.split_heads(query, key, value)
        query, key = self.apply_rotary(query, key, rotary_embeddings)
        output = self.attend_heads(query, key, value, cache=cache, attn_mask=attn_mask)
        return self.o_proj(output * torch.sigmoid(gate))


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.attn = TextAttention(config)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)
        self.norm_1 = ZCRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_2 = ZCRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None):
        residual = x
        hidden = self.norm_1(x)
        if self.layer_type == "linear_attention":
            hidden = self.linear_attn(hidden, cache=cache)
        else:
            hidden = self.attn(hidden, rotary_embeddings=(cos, sin), cache=cache)
        x = residual + hidden

        residual = x
        hidden = self.mlp(self.norm_2(x))
        return residual + hidden


class Qwen3_5Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_cache = getattr(config, "use_cache", False)
        self.layer_types = config.layer_types
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList([DecoderLayer(config, layer_index) for layer_index in range(config.num_hidden_layers)])
        self.norm = ZCRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope = RoPE(
            rotary_inv_freq(
                config.head_dim,
                config.rope_parameters["rope_theta"],
                int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)),
            )
        )
        if self.use_cache:
            self.caches = build_layers_cache(self.layer_types, config)
        else:
            self.caches = None
        self.position_offset = 0

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seq_len = inputs_embeds.shape[1]

        if position_ids is None:
            start = self.position_offset if self.use_cache else 0
            position_ids = torch.arange(start, start + seq_len, device=inputs_embeds.device)
        cos, sin = self.rope(position_ids.view(1, -1))

        for layer, cache in zip(self.layers, self.caches or [None] * len(self.layers)):
            inputs_embeds = layer(inputs_embeds, cos, sin, cache=cache)

        if self.use_cache:
            self.position_offset += seq_len
        return self.norm(inputs_embeds)

    def clear_cache(self):
        if self.caches is not None:
            for cache in self.caches:
                cache.clear()
        self.position_offset = 0
