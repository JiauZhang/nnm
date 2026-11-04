import torch
from torch import nn
from torch.nn import functional as F

from nnm.layers.mlp import GateMLP
from nnm.layers.moe import MoE
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import apply_rope
from nnm.models.unlimited_ocr.sparse import BlockRouter, MASK_NEG, SparseIndexer, sparse_keep_mask


class DecoderAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rms_norm_eps,
        layer_idx,
        sparse,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.block_size = int(sparse["sparse_block_size"])
        self.local_window = max(1, int(sparse["sparse_local_block"])) * self.block_size
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        index_dim = int(sparse["sparse_index_dim"])
        self.indexer = SparseIndexer(
            hidden_size=hidden_size,
            index_dim=index_dim,
            num_index_heads=int(sparse.get("sparse_num_index_heads", 1)),
            eps=rms_norm_eps,
        )
        scale_per_layer = sparse.get("dash_scale_per_layer")
        floor_per_layer = sparse.get("dash_floor_per_layer")
        self.dash_router = BlockRouter(
            hidden_size=hidden_size,
            index_dim=index_dim,
            eps=rms_norm_eps,
            block_size=self.block_size,
            local_blocks=max(1, int(sparse["sparse_local_block"])),
            topk_blocks=int(sparse.get("dash_topk_blocks", sparse["sparse_topk_blocks"])),
            scaling_factor=float(scale_per_layer[layer_idx] if scale_per_layer else sparse.get("dash_scaling_factor", 2.0)),
            sigma=float(sparse.get("dash_sigma", 1.0)),
            prior_floor=float(floor_per_layer[layer_idx] if floor_per_layer else sparse.get("dash_prior_floor", 1e-2)),
            stage0=str(sparse.get("dash_stage0", "indexer")).lower(),
            query_chunk=int(sparse.get("dash_query_chunk", 2048)),
            image_only=bool(sparse.get("dash_image_only", True)),
        )

    def forward(self, x, cos, sin, position_ids, block_candidate, image_blocks, attention_mask=None):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin, dim=1), apply_rope(k, cos, sin, dim=1)

        num_key_blocks = (seq_len + self.block_size - 1) // self.block_size
        block_scores = self.dash_router.block_scores(x, cos, sin, position_ids, num_key_blocks)
        top_indices, bias, valid, _, _ = self.dash_router.route_and_select(block_scores, block_candidate, image_blocks)

        keep = sparse_keep_mask(top_indices, position_ids, seq_len, self.block_size, self.local_window, valid)
        key_positions = torch.arange(seq_len, device=x.device)
        local = (key_positions[None, None, :] >= (position_ids - (self.local_window - 1)).clamp(min=0)[:, :, None]) & (
            key_positions[None, None, :] <= position_ids[:, :, None]
        )
        bias_map = torch.full((batch, seq_len, num_key_blocks), MASK_NEG, device=x.device, dtype=bias.dtype).scatter(
            -1, top_indices, torch.where(valid, bias, bias.new_full((), MASK_NEG))
        )
        token_bias = bias_map.index_select(-1, key_positions // self.block_size)[:, None]
        mask = torch.where(local, torch.zeros_like(token_bias), token_bias)
        mask = mask.masked_fill(~keep, MASK_NEG)
        if attention_mask is not None:
            mask = mask + attention_mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype), scale=self.scale, enable_gqa=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, *, config, layer_idx, sparse):
        super().__init__()
        hidden_size = config.hidden_size
        head_dim = hidden_size // config.num_attention_heads
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DecoderAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
            sparse=sparse,
        )
        use_moe = (
            config.get("n_routed_experts") is not None
            and layer_idx >= int(config.get("first_k_dense_replace", 0))
            and layer_idx % int(config.get("moe_layer_freq", 1)) == 0
        )
        if use_moe:
            self.mlp = MoE(
                hidden_size=hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                scoring_func=config.get("scoring_func", "softmax"),
                norm_topk_prob=bool(config.get("norm_topk_prob", False)),
                routed_scaling_factor=float(config.get("routed_scaling_factor", 1.0)),
                act=nn.SiLU,
            )
        else:
            self.mlp = GateMLP(hidden_size=hidden_size, intermediate_size=config.intermediate_size, act=nn.SiLU)

    def forward(self, x, cos, sin, position_ids, block_candidate, image_blocks, attention_mask=None):
        residual = x
        x = self.self_attn(self.input_layernorm(x), cos, sin, position_ids, block_candidate, image_blocks, attention_mask)
        x = residual + x
        residual = x
        return residual + self.mlp(self.post_attention_layernorm(x))
