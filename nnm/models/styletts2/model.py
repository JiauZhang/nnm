import torch
from torch import nn

from nnm.layers.diffusion.kdiffusion import KDiffusion
from nnm.layers.diffusion.schedule import LogNormalDistribution
from nnm.models.albert import AlbertModel
from nnm.models.pretrained import PretrainedModel

from .decoder import Decoder
from .diffusion import AudioDiffusionConditional, DiffusionTransformer1d, StyleTransformer1d
from .predictor import ProsodyPredictor
from .style_encoder import StyleEncoder
from .text_encoder import TextEncoder


def length_to_mask(lengths):
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


class StyleTTS2(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        style_dim = config.style_dim

        self.bert = AlbertModel(config.albert)
        self.bert_encoder = nn.Linear(config.albert.hidden_size, hidden_dim)
        self.text_encoder = TextEncoder(channels=hidden_dim, kernel_size=5, depth=config.n_layer, n_symbols=config.n_token)
        self.predictor = ProsodyPredictor(
            style_dim=style_dim, d_hid=hidden_dim, nlayers=config.n_layer, max_dur=config.max_dur, dropout=config.dropout
        )
        self.style_encoder = StyleEncoder(dim_in=config.dim_in, style_dim=style_dim, max_conv_dim=config.max_conv_dim)
        self.predictor_encoder = StyleEncoder(dim_in=config.dim_in, style_dim=style_dim, max_conv_dim=config.max_conv_dim)

        dec = config.decoder
        self.decoder = Decoder(
            dim_in=hidden_dim,
            style_dim=style_dim,
            dim_out=config.n_mels,
            resblock_kernel_sizes=dec.resblock_kernel_sizes,
            upsample_rates=dec.upsample_rates,
            upsample_initial_channel=dec.upsample_initial_channel,
            resblock_dilation_sizes=dec.resblock_dilation_sizes,
            upsample_kernel_sizes=dec.upsample_kernel_sizes,
        )

        diff = config.diffusion
        transformer_cls = StyleTransformer1d if config.multispeaker else DiffusionTransformer1d
        context_features = style_dim * 2 if config.multispeaker else None
        transformer = transformer_cls(
            channels=style_dim * 2,
            context_embedding_features=config.albert.hidden_size,
            context_features=context_features,
            num_layers=diff.transformer.num_layers,
            num_heads=diff.transformer.num_heads,
            head_features=diff.transformer.head_features,
            multiplier=diff.transformer.multiplier,
        )
        self.diffusion = AudioDiffusionConditional(
            embedding_features=config.albert.hidden_size,
            embedding_max_length=config.albert.max_position_embeddings,
            embedding_mask_proba=diff.embedding_mask_proba,
            channels=style_dim * 2,
            context_features=style_dim * 2,
            transformer=transformer,
        )
        self.diffusion.diffusion = KDiffusion(
            net=transformer,
            sigma_distribution=LogNormalDistribution(mean=diff.dist.mean, std=diff.dist.std),
            sigma_data=diff.dist.sigma_data,
            dynamic_threshold=0.0,
        )
        self.diffusion.unet = transformer

    @torch.no_grad()
    def encode_text(self, input_ids):
        n_tok = input_ids.shape[-1]
        il = torch.tensor([n_tok], dtype=torch.long, device=input_ids.device)
        tm = length_to_mask(il)
        t_en = self.text_encoder(input_ids, il, tm)
        bd = self.bert(input_ids, attention_mask=(~tm).int())
        d_en = self.bert_encoder(bd).transpose(-1, -2)
        return t_en, d_en, il, tm

    @torch.no_grad()
    def extract_style(self, mel):
        return self.style_encoder(mel), self.predictor_encoder(mel)

    @torch.no_grad()
    def sample_style(self, bert_embedding, reference=None, num_steps=5):
        noise = torch.randn((bert_embedding.shape[0], self.config.style_dim * 2), device=bert_embedding.device).unsqueeze(1)
        kwargs = {"embedding": bert_embedding, "embedding_scale": 1.0}
        if reference is not None:
            kwargs["features"] = reference
        return self.diffusion.diffusion.sample(noise, num_steps=num_steps, **kwargs).squeeze(1)

    @torch.no_grad()
    def infer(self, input_ids, style, speed=1.0):
        if style.dim() == 1:
            style = style.unsqueeze(0)
        sa, sp = style[:, : self.config.style_dim], style[:, self.config.style_dim :]
        t_en, d_en, il, tm = self.encode_text(input_ids)
        d = self.predictor.text_encoder(d_en, sp, il, tm)
        x, _ = self.predictor.lstm(d)
        dur = torch.sigmoid(self.predictor.duration_proj(x)).sum(-1) / float(speed)
        pr = torch.round(dur.reshape(-1)).clamp(min=1).long()
        n_tok = input_ids.shape[-1]
        aln = torch.repeat_interleave(torch.eye(n_tok, device=input_ids.device), pr, dim=1)
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        F0, N = self.predictor.F0Ntrain(en, sp)
        F0 = F0.unsqueeze(0) if F0.dim() == 1 else F0
        N = N.unsqueeze(0) if N.dim() == 1 else N
        waveform = self.decoder(t_en @ aln.unsqueeze(0), F0, N, sa)
        return {"waveform": waveform.reshape(-1), "duration": pr}
