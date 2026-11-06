import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from nnm.layers.rope import apply_rope

SCORE_NEG = -1.0e9
MASK_NEG = -1.0e9
ACT_CLAMP = 1.0e4


class PageIndexNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.clamp(-ACT_CLAMP, ACT_CLAMP)
        out = (x * (1.0 + self.weight.float())).clamp(-ACT_CLAMP, ACT_CLAMP)
        return out.to(dtype)


def entmax15(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    x = x / 2
    xsrt, _ = torch.sort(x, descending=True, dim=dim)
    rho = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype).view(*([1] * (x.dim() - 1)), -1)
    mean = xsrt.cumsum(dim) / rho
    mean_sq = (xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho
    tau = mean - torch.sqrt(torch.clamp(delta, min=0))
    support = (tau <= xsrt).sum(dim, keepdim=True).clamp(min=1)
    tau_star = tau.gather(dim, support - 1)
    return torch.clamp(x - tau_star, min=0) ** 2


class PageLayout(NamedTuple):
    image_mask: torch.Tensor | None
    image_pages: torch.Tensor | None
    query_pages: torch.Tensor | None
    rope_tokens: torch.Tensor | None
    rope_pages: torch.Tensor | None

    def image_blocks(self, num_key_blocks, block_size):
        batch, seq_len = self.image_mask.shape
        padded = num_key_blocks * block_size
        mask = self.image_mask
        if padded > seq_len:
            mask = F.pad(mask, (0, padded - seq_len), value=False)
        else:
            mask = mask[..., :padded]
        return mask.view(batch, num_key_blocks, block_size).any(dim=-1)

    def candidate_mask(self, pos_chunk, num_key_blocks, block_size):
        batch, chunk = pos_chunk.shape
        if self.image_pages is not None and self.query_pages is not None:
            page_len = self.image_pages.shape[-1]
            query_pages = self.query_pages.gather(1, pos_chunk.clamp(0, page_len - 1))
            padded = num_key_blocks * block_size - page_len
            if padded > 0:
                image_pages = F.pad(self.image_pages.float(), (0, padded), value=-1).long()
            else:
                image_pages = self.image_pages[..., : num_key_blocks * block_size]
            image_page_blocks = image_pages.view(batch, num_key_blocks, block_size)
            same_page = (query_pages[:, :, None, None] >= 0) & (
                query_pages[:, :, None, None] == image_page_blocks[:, None, :, :]
            )
            candidate = same_page.any(dim=-1).unsqueeze(1)
        elif self.image_mask is not None:
            blocks = self.image_blocks(num_key_blocks, block_size)
            candidate = blocks[:, None, None, :].expand(batch, 1, chunk, num_key_blocks).contiguous()
        else:
            candidate = torch.ones(batch, 1, chunk, num_key_blocks, dtype=torch.bool, device=pos_chunk.device)
        has_candidate = candidate.any(dim=-1, keepdim=True)
        return torch.where(has_candidate, candidate, torch.ones_like(candidate))


def build_page_layout(input_ids, image_token_id, page_token_id, per_page_image_tokens):
    if input_ids is None or image_token_id is None or page_token_id is None:
        return PageLayout(None, None, None, None, None)
    device = input_ids.device
    image_mask = input_ids == image_token_id
    has_image = bool(image_mask.any())
    marker_mask = input_ids == page_token_id
    positions = torch.arange(input_ids.shape[-1], device=device)
    last_image = (image_mask.long() * (positions + 1)).max(dim=-1).values - 1
    cum_pages = torch.cumsum(marker_mask.long(), dim=-1)
    n_pages = cum_pages.gather(1, last_image.clamp(min=0)[:, None]) * (last_image >= 0)[:, None].long()
    in_prefill = positions[None, :] <= last_image[:, None]

    per_page = max(1, int(per_page_image_tokens))
    running = torch.cumsum(image_mask.long(), dim=-1) - 1
    image_pages = torch.where(n_pages > 0, cum_pages - 1, running // per_page).masked_fill(~image_mask, -1)

    raw_page = cum_pages - 1
    response_page = raw_page - torch.where(n_pages > 0, n_pages, torch.zeros_like(n_pages))
    query_pages = torch.where(in_prefill, raw_page, response_page)
    query_pages = query_pages.masked_fill(query_pages < 0, -1)
    if not has_image or (query_pages >= 0).sum() == 0:
        image_mask = image_pages = query_pages = None

    if not (has_image and bool(marker_mask.any()) and bool((n_pages > 0).all())):
        rope_tokens = rope_pages = None
    else:
        page = torch.where(in_prefill, cum_pages, cum_pages - n_pages).clamp(min=0)
        onehot = F.one_hot(page, num_classes=int(page.max().item()) + 1).to(torch.int32)
        run = onehot.cumsum(dim=1)
        rope_tokens = run.gather(2, page[:, :, None].to(torch.int64)).squeeze(-1).to(torch.int64) - 1
        rope_pages = page.to(torch.int64)

    return PageLayout(image_mask, image_pages, query_pages, rope_tokens, rope_pages)


def sparse_keep_mask(block_indices, position_ids, key_length, block_size, local_window, valid):
    if block_indices.dim() == 3:
        block_indices = block_indices.unsqueeze(1)
        valid = valid.unsqueeze(1)
    batch, _, query_len, _ = block_indices.shape
    num_key_blocks = (key_length + block_size - 1) // block_size
    safe = block_indices.masked_fill(~valid, num_key_blocks)
    bitmap = torch.zeros(batch, 1, query_len, num_key_blocks + 1, dtype=torch.bool, device=block_indices.device)
    bitmap.scatter_(-1, safe, True)
    bitmap = bitmap[..., :num_key_blocks]
    block_keep = bitmap.repeat_interleave(block_size, dim=-1)[..., :key_length]
    key_positions = torch.arange(key_length, device=position_ids.device)
    local = (key_positions[None, None, :] >= (position_ids - (local_window - 1)).clamp(min=0)[:, :, None]) & (
        key_positions[None, None, :] <= position_ids[:, :, None]
    )
    causal = key_positions[None, None, :] <= position_ids[:, :, None]
    return (block_keep | local) & causal


class BlockRouter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size,
        index_dim,
        eps,
        block_size,
        local_blocks,
        topk_blocks,
        scaling_factor,
        sigma,
        prior_floor,
        stage0,
        query_chunk,
        image_only=True,
    ):
        super().__init__()
        self.index_dim = index_dim
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.local_window = local_blocks * block_size
        self.topk_blocks = topk_blocks
        self.scaling_factor = scaling_factor
        self.sigma = sigma
        self.prior_floor = prior_floor
        self.stage0 = stage0
        self.query_chunk = query_chunk
        self.image_only = image_only
        self.scale = scaling_factor / math.sqrt(index_dim)
        self.softmax_scale = index_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.q_norm = PageIndexNorm(index_dim, eps=eps)
        self.k_norm = PageIndexNorm(index_dim, eps=eps)
        self.head_cls = nn.Parameter(torch.zeros(index_dim))

    def project(self, hidden_states, cos, sin):
        batch, seq_len, _ = hidden_states.shape
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        idx_q = self.q_norm(self.q_proj(hidden_states).view(batch, seq_len, 1, self.index_dim)).transpose(1, 2)
        idx_k = self.k_norm(self.k_proj(hidden_states).view(batch, seq_len, 1, self.index_dim)).transpose(1, 2)
        idx_q, idx_k = apply_rope(idx_q, cos, sin, dim=1), apply_rope(idx_k, cos, sin, dim=1)
        return torch.nan_to_num(idx_q[:, 0].float()), torch.nan_to_num(idx_k[:, 0].float())

    def block_scores(self, hidden_states, cos, sin, position_ids, num_key_blocks):
        idx_q, idx_k = self.project(hidden_states, cos, sin)
        if self.stage0 == "head_cls":
            return self._head_cls_scores(idx_q, idx_k, position_ids, num_key_blocks)
        return self._indexer_scores(idx_q, idx_k, position_ids, num_key_blocks)

    def _head_cls_scores(self, idx_q, idx_k, position_ids, num_key_blocks):
        batch, seq_len, dim = idx_q.shape
        padded_len = num_key_blocks * self.block_size
        if padded_len > seq_len:
            idx_k = F.pad(idx_k, (0, 0, 0, padded_len - seq_len))
            valid = F.pad(idx_k.new_ones(batch, seq_len, dtype=torch.bool), (0, padded_len - seq_len), value=False)
        else:
            idx_k = idx_k[:, :padded_len]
            valid = idx_k.new_ones(batch, padded_len, dtype=torch.bool)
        blocks = idx_k.view(batch, num_key_blocks, self.block_size, dim)
        valid = valid.view(batch, num_key_blocks, self.block_size)
        inv_sqrt_dim = 1.0 / math.sqrt(dim)
        head = self.head_cls.to(blocks.dtype).view(1, 1, 1, dim)
        weights = ((blocks * head).sum(-1) * inv_sqrt_dim).clamp(-ACT_CLAMP, ACT_CLAMP)
        row_max = torch.where(valid, weights, weights.new_full((), torch.finfo(weights.dtype).min)).amax(dim=-1, keepdim=True)
        row_max = torch.where(valid.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max))
        weights = torch.exp(weights - row_max) * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        summary = torch.nan_to_num((weights.unsqueeze(-1) * blocks).sum(dim=2), nan=0.0, posinf=0.0, neginf=0.0)
        chunk_valid = valid.any(dim=-1)
        block_ids = torch.arange(num_key_blocks, device=idx_q.device)
        outputs = []
        for start in range(0, seq_len, self.query_chunk):
            end = min(start + self.query_chunk, seq_len)
            scores = torch.matmul(idx_q[:, start:end], summary.transpose(-1, -2)) * inv_sqrt_dim
            scores = scores.clamp(-ACT_CLAMP, ACT_CLAMP)
            query_block = position_ids[:, start:end] // self.block_size
            remote_ok = block_ids[None, None, :] <= (query_block[:, :, None] - self.local_blocks)
            scores = scores.masked_fill(~(remote_ok & chunk_valid[:, None, :]), SCORE_NEG)
            outputs.append(scores)
        return torch.cat(outputs, dim=1)

    def _indexer_scores(self, idx_q, idx_k, position_ids, num_key_blocks):
        batch, seq_len, _ = idx_q.shape
        key_len = idx_k.shape[1]
        key_positions = torch.arange(key_len, device=idx_q.device)
        padded_len = num_key_blocks * self.block_size
        outputs = []
        for start in range(0, seq_len, self.query_chunk):
            end = min(start + self.query_chunk, seq_len)
            scores = torch.matmul(idx_q[:, start:end], idx_k.transpose(-1, -2)) * self.softmax_scale
            query_pos = position_ids[:, start:end]
            future = key_positions[None, None, :] > query_pos[:, :, None]
            local = (key_positions[None, None, :] > (query_pos[:, :, None] - self.local_window)) & ~future
            scores = scores.masked_fill(future | local, SCORE_NEG)
            if padded_len > key_len:
                scores = F.pad(scores, (0, padded_len - key_len), value=SCORE_NEG)
            scores = scores.view(batch, end - start, num_key_blocks, self.block_size)
            outputs.append(scores.amax(dim=-1))
        return torch.cat(outputs, dim=1)

    def route_and_select(self, block_scores, block_candidate=None, image_blocks=None):
        batch, seq_len, num_key_blocks = block_scores.shape
        remote = block_scores > (SCORE_NEG * 0.5)
        candidate = remote
        if block_candidate is not None:
            candidate = remote & block_candidate
        elif self.image_only and image_blocks is not None:
            candidate = remote & image_blocks[:, None, :]
        candidate = torch.where(candidate.any(dim=-1, keepdim=True), candidate, remote)
        scores = torch.where(candidate, block_scores * self.scale, block_scores.new_full((), SCORE_NEG))
        scores = torch.nan_to_num(scores, nan=SCORE_NEG, posinf=SCORE_NEG, neginf=SCORE_NEG)
        prior = entmax15(scores, dim=-1)
        top_k = max(1, min(self.topk_blocks, num_key_blocks))
        ranked = prior.masked_fill(~candidate, -1.0)
        top_values, top_indices = ranked.topk(top_k, dim=-1)
        valid = top_values >= 0.0
        log_prior = torch.log(top_values.clamp_min(self.prior_floor))
        count = valid.sum(dim=-1, keepdim=True).clamp(min=1)
        mean = torch.where(valid, log_prior, torch.zeros_like(log_prior)).sum(dim=-1, keepdim=True) / count
        bias = torch.where(valid, (log_prior - mean) / self.sigma, torch.zeros_like(log_prior)).clamp(-ACT_CLAMP, ACT_CLAMP)
        return top_indices, bias, valid, prior, candidate
