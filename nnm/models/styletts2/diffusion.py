import math

import torch
import torch.nn.functional as F
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


class AttentionBase(nn.Module):
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


class Attention(nn.Module):
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
        self.attention = AttentionBase(
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
        self.attention = AttentionBase(
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


class TransformerBlock(nn.Module):
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
        self.attention = Attention(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        if self.use_cross_attention:
            self.cross_attention = Attention(
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


class FixedEmbedding(nn.Module):
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
        self.fixed_embedding = FixedEmbedding(embedding_max_length, context_embedding_features)

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


class Transformer1d(_Transformer1dBase):
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
                TransformerBlock(
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


class LogNormalDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, num_samples, device=torch.device("cpu")):
        return (self.mean + self.std * torch.randn((num_samples,), device=device)).exp()


class UniformDistribution:
    def __call__(self, num_samples, device=torch.device("cpu")):
        return torch.rand(num_samples, device=device)


def _to_batch(batch_size, device, x=None, xs=None):
    if x is not None:
        return torch.full((batch_size,), fill_value=x, device=device)
    return xs


class KDiffusion(nn.Module):
    def __init__(self, net, *, sigma_distribution, sigma_data, dynamic_threshold=0.0):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas):
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = sigmas[:, None, None]
        c_skip = (sigma_data**2) / (sigmas**2 + sigma_data**2)
        c_out = sigmas * sigma_data * (sigma_data**2 + sigmas**2) ** -0.5
        c_in = (sigmas**2 + sigma_data**2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(self, x_noisy, sigmas=None, sigma=None, **kwargs):
        sigmas = _to_batch(x_noisy.shape[0], x_noisy.device, x=sigma, xs=sigmas)
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        return c_skip * x_noisy + c_out * x_pred

    def loss_weight(self, sigmas):
        return (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x, noise=None, **kwargs):
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        noise = torch.randn_like(x) if noise is None else noise
        x_noisy = x + sigmas[:, None, None] * noise
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)
        losses = F.mse_loss(x_denoised, x, reduction="none").flatten(1).mean(1)
        return (losses * self.loss_weight(sigmas)).mean()

    def sample(self, noise, num_steps=5, **kwargs):
        sampler = DiffusionSampler(
            self, sampler=ADPM2Sampler(), sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False
        )
        return sampler(noise, num_steps=num_steps, **kwargs)


class LinearSchedule(nn.Module):
    def forward(self, num_steps, device):
        return torch.linspace(1, 0, num_steps + 1)[:-1]


class KarrasSchedule(nn.Module):
    def __init__(self, sigma_min, sigma_max, rho=7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps, device):
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**rho_inv + (steps / (num_steps - 1)) * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return F.pad(sigmas, pad=(0, 1), value=0.0)


class VSampler(nn.Module):
    def get_alpha_beta(self, sigma):
        angle = sigma * math.pi / 2
        return math.cos(angle), math.sin(angle)

    def forward(self, noise, fn, sigmas, num_steps):
        x = sigmas[0] * noise
        alpha, beta = self.get_alpha_beta(sigmas[0].item())
        for i in range(num_steps - 1):
            is_last = i == num_steps - 1
            x_denoised = fn(x, sigma=sigmas[i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha
            if not is_last:
                alpha, beta = self.get_alpha_beta(sigmas[i + 1].item())
                x = x_pred * alpha + x_eps * beta
        return x_pred


class KarrasSampler(nn.Module):
    def __init__(self, s_tmin=0, s_tmax=float("inf"), s_churn=0.0, s_noise=1.0):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(self, x, fn, sigma, sigma_next, gamma):
        sigma_hat = sigma + gamma * sigma
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + math.sqrt(sigma_hat**2 - sigma**2) * epsilon
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(self, noise, fn, sigmas, num_steps):
        x = sigmas[0] * noise
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax), min(self.s_churn / num_steps, math.sqrt(2) - 1), 0.0
        )
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i])
        return x


class ADPM2Sampler(nn.Module):
    def __init__(self, rho=1.0):
        super().__init__()
        self.rho = rho

    def get_sigmas(self, sigma, sigma_next):
        r = self.rho
        sigma_up = math.sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2)
        sigma_down = math.sqrt(sigma_next**2 - sigma_up**2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x, fn, sigma, sigma_next):
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        d = (x - fn(x, sigma=sigma)) / sigma
        x_mid = x + d * (sigma_mid - sigma)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)
        return x + torch.randn_like(x) * sigma_up

    def forward(self, noise, fn, sigmas, num_steps):
        x = sigmas[0] * noise
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])
        return x


class DiffusionSampler(nn.Module):
    def __init__(self, diffusion, *, sampler, sigma_schedule, num_steps=None, clamp=True):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

    def forward(self, noise, num_steps=None, **kwargs):
        num_steps = self.num_steps if num_steps is None else num_steps
        sigmas = self.sigma_schedule(num_steps, noise.device)

        def fn(*a, **ka):
            return self.denoise_fn(*a, **{**ka, **kwargs})

        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        return x.clamp(-1.0, 1.0) if self.clamp else x


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
