import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.attention_mask import make_causal_mask
from nnm.layers.embedding import merge_placeholders
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.models.lfm2 import Lfm2DecoderLayer
from nnm.models.lfm3.language import Attention, DecoderLayer, Lfm3EmbeddingBackbone
from nnm.models.pretrained import PretrainedModel


class AudioDownsampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, channels, **conv)
        self.conv_mid = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_mid = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_end = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_end = nn.Conv2d(channels, channels, kernel_size=1)
        self.lin_out = nn.Conv1d(config.num_mel_bins * channels // 8, config.hidden_size, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, input_features, feature_lengths):
        def mask_padding(hidden_states, lengths):
            if lengths is None:
                return hidden_states
            valid = torch.arange(hidden_states.shape[-1], device=hidden_states.device)[None] < lengths[:, None]
            return hidden_states * valid[:, None, None].to(hidden_states.dtype)

        hidden = mask_padding(input_features.unsqueeze(1), feature_lengths)
        hidden = self.act(self.conv_in(hidden))
        lengths = None if feature_lengths is None else (feature_lengths + 1) // 2
        hidden = mask_padding(hidden, lengths)
        hidden = self.act(self.lin_mid(self.conv_mid(hidden)))
        lengths = None if lengths is None else (lengths + 1) // 2
        hidden = mask_padding(hidden, lengths)
        hidden = self.act(self.lin_end(self.conv_end(hidden)))
        return self.lin_out(hidden.flatten(1, 2))

    @staticmethod
    def output_lengths(lengths):
        return -(lengths // -8)


class AudioEncoderTrunk(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        base = config.rope_parameters["rope_theta"]
        self.layer_types = config.layer_types
        self.position_encoder = RoPE(rotary_inv_freq(head_dim, base))
        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(
                    config=config,
                    layer_idx=index,
                    is_attention_layer=kind == "full_attention",
                    head_dim=head_dim,
                )
                for index, kind in enumerate(config.layer_types)
            ]
        )
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=True)

    def forward(self, inputs_embeds, sliding_window):
        seq_len = inputs_embeds.shape[1]
        mask = make_causal_mask(seq_len, seq_len, 0, inputs_embeds.device, inputs_embeds.dtype, sliding_window=sliding_window)
        positions = torch.arange(seq_len, device=inputs_embeds.device)
        rotary_embeddings = self.position_encoder(positions)
        hidden = inputs_embeds
        for kind, layer in zip(self.layer_types, self.layers):
            hidden = layer(
                hidden,
                rotary_embeddings=rotary_embeddings,
                attn_mask=mask if kind == "full_attention" else None,
            )
        return self.embedding_norm(hidden)


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.downsample = AudioDownsampler(config)
        self.lfm = AudioEncoderTrunk(config.text_config)

    def forward(self, input_features, feature_lengths):
        hidden = self.downsample(input_features, feature_lengths).transpose(1, 2)
        encoded = self.lfm(hidden, self.config.sliding_window)
        return encoded, self.downsample.output_lengths(feature_lengths)


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_features = config.audio_encoder_config.hidden_size
        out_features = config.text_config.hidden_size
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(out_features, out_features)

    def forward(self, audio_features):
        return self.linear_2(self.act(self.linear_1(self.layer_norm(audio_features))))


class CodebookEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.depth_config.hidden_size)
        self.embedding_norm = RMSNorm(config.depth_config.hidden_size, config.depth_config.norm_eps)
        self.to_logits = nn.Linear(config.depth_config.hidden_size, config.codebook_size, bias=False)

    def forward(self, codes):
        return self.embedding(codes)

    def logits(self, hidden_states):
        return self.to_logits(self.embedding_norm(hidden_states))


class DepthAttention(Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.causal = False


class DepthLayer(DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = DepthAttention(config, layer_idx)


class DepthTrunk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rotary = nn.ModuleDict(
            {
                layer_type: RoPE(
                    rotary_inv_freq(
                        config.head_dim,
                        config.rope_parameters[layer_type]["rope_theta"],
                        int(config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0) * config.head_dim),
                    )
                )
                for layer_type in config.rope_parameters
            }
        )
        self.layers = nn.ModuleList([DepthLayer(config, index) for index in range(config.num_hidden_layers)])
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, inputs_embeds):
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            cos, sin = self.rotary[self.config.layer_types[index]](positions)
            hidden = layer(hidden, cos, sin)
        return self.embedding_norm(hidden)


class AudioDepthDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.hidden_size = config.depth_config.hidden_size
        self.depth_linear = nn.Linear(config.text_config.hidden_size, self.num_codebooks * self.hidden_size)
        self.mask_embedding = nn.Parameter(torch.empty(self.hidden_size))
        self.model = DepthTrunk(config.depth_config)
        self.depth_embeddings = nn.ModuleList(CodebookEmbedding(config) for _ in range(self.num_codebooks))

    def embed_codes(self, codes):
        return torch.stack([embedding(codes[..., index]) for index, embedding in enumerate(self.depth_embeddings)], dim=-2)

    def forward(self, hidden_states, codes, masked):
        depth_in = self.depth_linear(hidden_states).unflatten(-1, (self.num_codebooks, self.hidden_size))
        code_embeds = self.embed_codes(codes)
        depth_in = depth_in + torch.where(masked[..., None], self.mask_embedding.to(code_embeds.dtype), code_embeds)
        depth_out = self.model(depth_in)
        return torch.stack(
            [embedding.logits(depth_out[:, index]) for index, embedding in enumerate(self.depth_embeddings)], dim=1
        )


def audio_codebook_loss(depth_decoder, hidden_states, input_ids, audio_labels, config):
    placeholders = input_ids == config.audio_out_token_id
    preceding = placeholders.clone()
    preceding[:, 0] = False
    supervised = (audio_labels != -100).all(-1)[preceding[placeholders]]
    latents = hidden_states[preceding.roll(-1, dims=1)][supervised]
    codes = audio_labels[preceding[placeholders]][supervised]
    if codes.shape[0] == 0:
        return hidden_states.sum() * 0.0
    ratio = torch.rand(codes.shape[0], 1, device=codes.device)
    masked = torch.rand(codes.shape, device=codes.device) < ratio
    masked |= ~masked.any(-1, keepdim=True)
    masked |= (codes == config.audio_eos_code).all(-1, keepdim=True)
    logits = depth_decoder(latents, codes, masked)
    losses = F.cross_entropy(logits.transpose(1, 2).float(), codes, reduction="none")
    return (losses * masked).sum() / masked.sum().clamp_min(1)


class Lfm3AudioModel(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.audio_token_id = config.audio_token_id
        self.audio_out_token_id = config.audio_out_token_id
        self.backbone = Lfm3EmbeddingBackbone(config.text_config)
        self.audio_encoder = AudioEncoder(config.audio_encoder_config)
        self.audio_projector = AudioProjector(config)
        self.codebook_embedding = nn.Embedding(config.num_codebooks * config.codebook_size, config.text_config.hidden_size)
        self.depth_decoder = AudioDepthDecoder(config)
        self.lm_head = (
            None
            if self.tie_word_embeddings
            else nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        )

    def get_audio_features(self, input_features, feature_attention_mask):
        feature_lengths = feature_attention_mask.sum(-1)
        encoded, encoded_lengths = self.audio_encoder(input_features, feature_lengths)
        valid = torch.arange(encoded.shape[1], device=encoded.device)[None] < encoded_lengths[:, None]
        return self.audio_projector(encoded[valid])

    def get_audio_out_embeds(self, audio_out_codes):
        codebook_size = self.config.codebook_size
        offsets = torch.arange(self.config.num_codebooks, device=audio_out_codes.device) * codebook_size
        return self.codebook_embedding(audio_out_codes + offsets).sum(-2)

    def embed_audio(self, inputs_embeds, input_ids, features, token_id):
        return merge_placeholders(inputs_embeds, input_ids == token_id, features)

    def embed_inputs(self, input_ids, input_features=None, feature_attention_mask=None, audio_out_codes=None):
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        if input_features is not None:
            if feature_attention_mask is None:
                feature_attention_mask = torch.ones(
                    input_features.shape[0], input_features.shape[-1], dtype=torch.bool, device=input_features.device
                )
            features = self.get_audio_features(input_features, feature_attention_mask)
            inputs_embeds = self.embed_audio(inputs_embeds, input_ids, features, self.audio_token_id)
        if audio_out_codes is not None:
            features = self.get_audio_out_embeds(audio_out_codes)
            inputs_embeds = self.embed_audio(inputs_embeds, input_ids, features, self.audio_out_token_id)
        return inputs_embeds

    def forward(
        self,
        input_ids,
        input_features=None,
        feature_attention_mask=None,
        audio_out_codes=None,
        attn_mask=None,
    ):
        inputs_embeds = self.embed_inputs(input_ids, input_features, feature_attention_mask, audio_out_codes)
        hidden = self.backbone(inputs_embeds=inputs_embeds, attn_mask=attn_mask)
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return torch.matmul(hidden, self.backbone.embed_tokens.weight.t())

    def audio_loss(self, hidden_states, input_ids, audio_labels):
        return audio_codebook_loss(self.depth_decoder, hidden_states, input_ids, audio_labels, self.config)

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()
        self.backbone.position_offset = 0
