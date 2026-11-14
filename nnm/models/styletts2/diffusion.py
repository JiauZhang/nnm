import math

import torch
from torch import nn

from .blocks import AdaLayerNorm


def FeedForward(features, multiplier):
    mid_features = features * multiplier
    return nn.Sequential(nn.Linear(features, mid_features), nn.GELU(), nn.Linear(mid_features, features))


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets, max_distance, num_heads):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = (
            max_exact
            + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return ret + torch.where(is_small, n, val_if_large)

    def forward(self, num_queries, num_keys):
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(num_keys - num_queries, num_keys, dtype=torch.long, device=device)
        k_pos = torch.arange(num_keys, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        bucket = self._relative_position_bucket(rel_pos, self.num_buckets, self.max_distance)
        bias = self.relative_attention_bias(bucket)
        return bias.permute(2, 0, 1)[None]


class DiffusionAttentionBase(nn.Module):
    def __init__(
        self,
        features,
        *,
        head_features,
        num_heads,
        use_rel_pos=False,
        out_features=None,
        rel_pos_num_buckets=None,
        rel_pos_max_distance=None,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        self.head_features = head_features
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads
        if use_rel_pos:
            self.rel_pos = RelativePositionBias(rel_pos_num_buckets, rel_pos_max_distance, num_heads)
        self.to_out = nn.Linear(mid_features, features if out_features is None else out_features)

    def forward(self, q, k, v):
        b, n, _ = q.shape
        m = k.shape[1]
        q = q.view(b, n, self.num_heads, self.head_features).transpose(1, 2)
        k = k.view(b, m, self.num_heads, self.head_features).transpose(1, 2)
        v = v.view(b, m, self.num_heads, self.head_features).transpose(1, 2)
        sim = torch.einsum("bhnd,bhmd->bhnm", q, k)
        if self.use_rel_pos:
            sim = sim + self.rel_pos(n, m)
        attn = (sim * self.scale).softmax(dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class DiffusionAttention(nn.Module):
    def __init__(
        self,
        features,
        *,
        head_features,
        num_heads,
        use_rel_pos=False,
        out_features=None,
        context_features=None,
        rel_pos_num_buckets=None,
        rel_pos_max_distance=None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = features if context_features is None else context_features
        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(features, mid_features, bias=False)
        self.to_kv = nn.Linear(context_features, mid_features * 2, bias=False)
        self.attention = DiffusionAttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x, *, context=None):
        context = x if context is None else context
        x, context = self.norm(x), self.norm_context(context)
        q = self.to_q(x)
        k, v = torch.chunk(self.to_kv(context), chunks=2, dim=-1)
        return self.attention(q, k, v)


class StyleAttention(nn.Module):
    def __init__(
        self,
        features,
        *,
        style_dim,
        head_features,
        num_heads,
        context_features=None,
        use_rel_pos=False,
        rel_pos_num_buckets=None,
        rel_pos_max_distance=None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = features if context_features is None else context_features
        self.norm = AdaLayerNorm(style_dim, features)
        self.norm_context = AdaLayerNorm(style_dim, context_features)
        self.to_q = nn.Linear(features, mid_features, bias=False)
        self.to_kv = nn.Linear(context_features, mid_features * 2, bias=False)
        self.attention = DiffusionAttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x, s, *, context=None):
        context = x if context is None else context
        x, context = self.norm(x, s), self.norm_context(context, s)
        q = self.to_q(x)
        k, v = torch.chunk(self.to_kv(context), chunks=2, dim=-1)
        return self.attention(q, k, v)


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        features,
        num_heads,
        head_features,
        multiplier,
        use_rel_pos=False,
        rel_pos_num_buckets=None,
        rel_pos_max_distance=None,
        context_features=None,
    ):
        super().__init__()
        self.use_cross_attention = context_features is not None and context_features > 0
        self.attention = DiffusionAttention(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        if self.use_cross_attention:
            self.cross_attention = DiffusionAttention(
                features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )
        self.feed_forward = FeedForward(features, multiplier)

    def forward(self, x, *, context=None):
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        return self.feed_forward(x) + x


class StyleTransformerBlock(nn.Module):
    def __init__(
        self,
        features,
        num_heads,
        head_features,
        style_dim,
        multiplier,
        use_rel_pos=False,
        rel_pos_num_buckets=None,
        rel_pos_max_distance=None,
        context_features=None,
    ):
        super().__init__()
        self.use_cross_attention = context_features is not None and context_features > 0
        self.attention = StyleAttention(
            features,
            style_dim=style_dim,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        if self.use_cross_attention:
            self.cross_attention = StyleAttention(
                features,
                style_dim=style_dim,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )
        self.feed_forward = FeedForward(features, multiplier)

    def forward(self, x, s, *, context=None):
        x = self.attention(x, s) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, s, context=context) + x
        return self.feed_forward(x) + x


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


def TimePositionalEmbedding(dim, out_features):
    return nn.Sequential(LearnedPositionalEmbedding(dim), nn.Linear(dim + 1, out_features))


class DiffusionFixedEmbedding(nn.Module):
    def __init__(self, max_length, features):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x):
        batch_size, length = x.shape[0], x.shape[1]
        position = torch.arange(length, device=x.device)
        return self.embedding(position)[None].expand(batch_size, -1, -1)


class _Transformer1dBase(nn.Module):
    def _init_context(self, channels, context_features, context_embedding_features, use_context_time, embedding_max_length):
        self.use_context_features = context_features is not None
        self.use_context_time = use_context_time
        if use_context_time or self.use_context_features:
            context_mapping_features = channels + context_embedding_features
            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )
        if use_context_time:
            self.to_time = nn.Sequential(TimePositionalEmbedding(channels, context_mapping_features), nn.GELU())
        if self.use_context_features:
            self.to_features = nn.Sequential(nn.Linear(context_features, context_mapping_features), nn.GELU())
        self.fixed_embedding = DiffusionFixedEmbedding(embedding_max_length, context_embedding_features)

    def get_mapping(self, time=None, features=None):
        items = []
        if self.use_context_time:
            items.append(self.to_time(time))
        if self.use_context_features:
            items.append(self.to_features(features))
        if items:
            return self.to_mapping(torch.stack(items).sum(0))
        return None

    def _run(self, x, time, embedding, features, block_fn):
        mapping = self.get_mapping(time, features)
        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], axis=-1)
        mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)
        for block in self.blocks:
            x = x + mapping
            x = block_fn(block, x, features)
        x = x.mean(axis=1).unsqueeze(1).transpose(1, 2)
        x = self.to_out(x)
        return x.transpose(-1, -2)

    def forward(self, x, time, embedding_mask_proba=0.0, embedding=None, features=None, embedding_scale=1.0):
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            batch_mask = torch.rand((b, 1, 1), device=device) < embedding_mask_proba
            embedding = torch.where(batch_mask, fixed_embedding, embedding)
        if embedding_scale != 1.0:
            out = self.run(x, time, embedding=embedding, features=features)
            out_masked = self.run(x, time, embedding=fixed_embedding, features=features)
            return out_masked + (out - out_masked) * embedding_scale
        return self.run(x, time, embedding=embedding, features=features)


class DiffusionTransformer1d(_Transformer1dBase):
    def __init__(
        self,
        num_layers,
        channels,
        num_heads,
        head_features,
        multiplier,
        use_context_time=True,
        use_rel_pos=False,
        rel_pos_num_buckets=None,
        rel_pos_max_distance=None,
        context_features=None,
        context_embedding_features=None,
        embedding_max_length=512,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    channels + context_embedding_features,
                    num_heads,
                    head_features,
                    multiplier,
                    use_rel_pos,
                    rel_pos_num_buckets,
                    rel_pos_max_distance,
                )
                for _ in range(num_layers)
            ]
        )
        self.to_out = nn.Conv1d(channels + context_embedding_features, channels, 1)
        self._init_context(channels, context_features, context_embedding_features, use_context_time, embedding_max_length)

    def run(self, x, time, embedding, features):
        return self._run(x, time, embedding, features, lambda block, x, feats: block(x))


class StyleTransformer1d(_Transformer1dBase):
    def __init__(
        self,
        num_layers,
        channels,
        num_heads,
        head_features,
        multiplier,
        use_context_time=True,
        use_rel_pos=False,
        rel_pos_num_buckets=None,
        rel_pos_max_distance=None,
        context_features=None,
        context_embedding_features=None,
        embedding_max_length=512,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                StyleTransformerBlock(
                    channels + context_embedding_features,
                    num_heads,
                    head_features,
                    context_features,
                    multiplier,
                    use_rel_pos,
                    rel_pos_num_buckets,
                    rel_pos_max_distance,
                )
                for _ in range(num_layers)
            ]
        )
        self.to_out = nn.Conv1d(channels + context_embedding_features, channels, 1)
        self._init_context(channels, context_features, context_embedding_features, use_context_time, embedding_max_length)

    def run(self, x, time, embedding, features):
        return self._run(x, time, embedding, features, lambda block, x, feats: block(x, feats))


class AudioDiffusionConditional(nn.Module):
    def __init__(
        self,
        embedding_features,
        embedding_max_length,
        embedding_mask_proba=0.1,
        channels=128,
        context_features=None,
        transformer=None,
    ):
        super().__init__()
        self.embedding_mask_proba = embedding_mask_proba
        self.unet = transformer
        self.diffusion = None

    def forward(self, x, **kwargs):
        return self.diffusion(x, embedding_mask_proba=self.embedding_mask_proba, **kwargs)

    def sample(self, *args, **kwargs):
        return self.diffusion.sample(*args, **kwargs)
