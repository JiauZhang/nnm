import math

import torch
import torch.nn.functional as F
from torch import nn

from nnm.backends.sdpa import scaled_dot_product_attention
from nnm.layers.dit import TimestepEmbedder, modulate
from nnm.layers.norm import WeightOnlyLayerNorm
from nnm.layers.rope import FusedQKVRoPE, apply_rope
from nnm.pretrained import PretrainedModel


def block_diff_mask(q_idx, kv_idx, block_size, n):
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n
    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
    return block_diagonal | offset_block_causal | block_causal


def bias_dropout_add_scale(x, bias, scale, residual, prob, training):
    out = x + bias if bias is not None else x
    out = scale * F.dropout(out, p=prob, training=training)
    return residual + out if residual is not None else out


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_dim, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDiTBlock(nn.Module):
    def __init__(self, n, block_size, dim, n_heads, cond_dim, causal=False, mlp_ratio=4, dropout=0.1, adaln=True):
        super().__init__()
        self.n = n
        self.block_size = block_size
        self.n_heads = n_heads
        self.kv_cache = None
        self.cache_idx = 0
        self.causal = causal
        self.dropout = dropout
        self.adaln = adaln

        self.norm1 = WeightOnlyLayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = WeightOnlyLayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        if self.adaln:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
            nn.init.zeros_(self.adaLN_modulation.weight)
            nn.init.zeros_(self.adaLN_modulation.bias)

    def get_qkv(self, x, rotary_cos_sin, store_kv=False):
        if self.kv_cache is not None:
            new_qkv = self.attn_qkv(x)
            self.kv_cache[:, self.cache_idx : self.cache_idx + self.block_size] = new_qkv
            qkv = self.kv_cache[:, : self.cache_idx + self.block_size].clone()
        else:
            qkv = self.attn_qkv(x)
        if store_kv:
            self.cache_idx += self.block_size
            if self.cache_idx >= self.n:
                self.cache_idx = self.n - self.block_size
                self.kv_cache[:, : -self.block_size] = self.kv_cache[:, self.block_size :].clone()
        qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.n_heads, -1)
        cos, sin = rotary_cos_sin
        return apply_rope(qkv, cos, sin)

    def cross_attn(self, qkv, mask=None):
        head_dim = qkv.shape[-1]
        qkv = qkv.transpose(1, 3)
        mask = mask.bool() if mask is not None else None
        x = scaled_dot_product_attention(
            qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], attn_mask=mask, is_causal=self.causal, scale=1 / math.sqrt(head_dim)
        )
        x = x.transpose(1, 2)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, x, rotary_cos_sin, c, mask=None, sample_mode=False, store_kv=False):
        if self.adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa) if self.adaln else self.norm1(x)

        if mask is not None and not sample_mode:
            n = mask.shape[-1] // 2
            qkv_x = self.get_qkv(x[:, :n], rotary_cos_sin)
            qkv_x0 = self.get_qkv(x[:, n:], rotary_cos_sin)
            qkv = torch.cat((qkv_x, qkv_x0), dim=1)
        else:
            qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)

        x = self.cross_attn(qkv, mask=mask)
        if self.kv_cache is not None:
            x = x[:, -self.block_size :]

        if self.adaln:
            x = bias_dropout_add_scale(self.attn_out(x), None, gate_msa, x_skip, self.dropout, self.training)
            x = bias_dropout_add_scale(
                self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout, self.training
            )
        else:
            x = bias_dropout_add_scale(self.attn_out(x), None, torch.ones_like(x), x_skip, self.dropout, self.training)
            x = bias_dropout_add_scale(self.mlp(self.norm2(x)), None, torch.ones_like(x), x, self.dropout, self.training)
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaln=True):
        super().__init__()
        self.norm_final = WeightOnlyLayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.adaln = adaln
        if self.adaln:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            nn.init.zeros_(self.adaLN_modulation.weight)
            nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c):
        if self.adaln:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        return self.linear(x)


class DITBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attn = config.cross_attn
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.n = config.model_length
        self.adaln = config.adaln
        self.time_conditioning = config.time_conditioning

        self.vocab_embed = EmbeddingLayer(config.hidden_size, config.vocab_size)
        if self.adaln:
            self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = FusedQKVRoPE(config.hidden_size // config.num_attention_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    self.n,
                    self.block_size,
                    config.hidden_size,
                    config.num_attention_heads,
                    config.cond_dim,
                    causal=config.causal,
                    dropout=config.dropout,
                    adaln=config.adaln,
                    mlp_ratio=config.intermediate_size // config.hidden_size,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.output_layer = DDitFinalLayer(config.hidden_size, config.vocab_size, config.cond_dim, adaln=config.adaln)
        if self.cross_attn:
            self.gen_mask(config.model_length, self.block_size)

    def gen_mask(self, seqlen, block_size):
        self.mask = block_diff_mask(
            torch.arange(seqlen * 2)[:, None], torch.arange(seqlen * 2)[None, :], block_size=block_size, n=seqlen
        )

    def forward(self, indices, sigma, sample_mode=False, store_kv=False, output_hidden_states=False):
        if not self.time_conditioning and self.adaln:
            sigma = torch.zeros_like(sigma)
        all_hidden_states = []
        x = self.vocab_embed(indices)
        if output_hidden_states:
            all_hidden_states.append(x)
        c = F.silu(self.sigma_map(sigma)) if self.adaln else None
        if self.cross_attn:
            mask = self.mask.to(x.device)
            n = self.mask.shape[-1] // 2
            if not sample_mode:
                rotary_cos_sin = self.rotary_emb(x[:, : self.n])
            else:
                if self.blocks[0].kv_cache is not None:
                    mask = None
                    accum_length = self.blocks[0].cache_idx + self.block_size
                    x_full = torch.zeros(x.shape[0], accum_length, x.shape[2], device=x.device)
                    rotary_cos_sin = self.rotary_emb(x_full)
                else:
                    rotary_cos_sin = self.rotary_emb(x[:, :n])
                    mask = mask[n : n + x.shape[1], n : n + x.shape[1]]
        else:
            mask = None
            rotary_cos_sin = self.rotary_emb(x)

        for block in self.blocks:
            x = block(x, rotary_cos_sin, c, mask=mask, sample_mode=sample_mode, store_kv=store_kv)
            if output_hidden_states:
                all_hidden_states.append(x)
        logits = self.output_layer(x, c)
        if self.cross_attn and not sample_mode:
            logits = logits[:, :n]
            all_hidden_states = [hidden[:, :n] for hidden in all_hidden_states]
        return logits, all_hidden_states


class Bd3lm(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = DITBackbone(config)
        if config.var_min:
            self.register_buffer("sampling_eps_min", torch.tensor(config.sampling_eps_min))
            self.register_buffer("sampling_eps_max", torch.tensor(config.sampling_eps_max))

    def reset_kv_cache(self, batch_size=1):
        for block in self.backbone.blocks:
            block.kv_cache = torch.zeros(
                batch_size,
                self.config.model_length,
                self.config.hidden_size * 3,
                device=self.backbone.vocab_embed.embedding.device,
                dtype=torch.float32,
            )
            block.cache_idx = 0

    def forward(self, input_ids, timesteps, sample_mode=False, store_kv=False, output_hidden_states=False):
        logits, all_hidden_states = self.backbone(
            indices=input_ids,
            sigma=timesteps,
            sample_mode=sample_mode,
            store_kv=store_kv,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            return logits, all_hidden_states
        return logits
