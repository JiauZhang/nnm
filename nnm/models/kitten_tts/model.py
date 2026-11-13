import torch
from torch import nn

from nnm.models.albert import AlbertModel
from nnm.models.pretrained import PretrainedModel

from .decoder import Decoder
from ..styletts2.predictor import ProsodyPredictor
from ..styletts2.text_encoder import TextEncoder

END_TOKEN = 10


def length_to_mask(lengths):
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def tokens_from_phonemes(phonemes, vocab):
    ids = []
    for ch in phonemes:
        if ch in vocab:
            ids.append(vocab[ch])
        elif ch in "()":
            ids.append(vocab[","])
    return [0] + ids + [END_TOKEN, 0]


class KittenTTS(PretrainedModel):
    module_name = "kitten_tts"

    def __init__(self, config, variant=None):
        super().__init__()
        self.config = config
        self.variant = variant
        hidden_dim = config.hidden_dim
        style_dim = config.style_dim

        self.bert = AlbertModel(config.albert)
        self.text_encoder = TextEncoder(
            channels=hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )
        self.bert_encoder = nn.Linear(config.albert.hidden_size, hidden_dim)
        self.predictor = ProsodyPredictor(
            style_dim=style_dim, d_hid=hidden_dim, nlayers=config.n_layer, max_dur=config.max_dur, dropout=config.dropout
        )
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
            gen_istft_n_fft=dec.gen_istft_n_fft,
            gen_istft_hop_size=dec.gen_istft_hop_size,
            encode_out_channels=dec.encode_out_channels,
            decode_out_channels=dec.decode_out_channels,
            decode_upsample_out_channels=dec.decode_upsample_out_channels,
            asr_res_in_channels=dec.asr_res_in_channels,
            adain_affine=dec.adain_affine,
            magnitude_activation=dec.magnitude_activation,
            training_downsample=dec.get("training_downsample", False),
            window_float64=dec.get("window_float64", False),
        )

    @classmethod
    def manifest_key(cls, **kwargs):
        variant = kwargs.get("variant")
        return f"{cls.module_name}_{variant}" if variant else cls.module_name

    @torch.no_grad()
    def infer(self, input_ids, style, speed=1.0):
        device = input_ids.device
        if style.dim() == 1:
            style = style.unsqueeze(0)
        sa, sp = style[:, :128], style[:, 128:]
        n_tok = input_ids.shape[-1]
        il = torch.tensor([n_tok], dtype=torch.long, device=device)
        tm = length_to_mask(il)

        t_en = self.text_encoder(input_ids, il, tm)
        bd = self.bert(input_ids, attention_mask=(~tm).int())
        d_en = self.bert_encoder(bd).transpose(-1, -2)
        d = self.predictor.text_encoder(d_en, sp, il, tm)
        x, _ = self.predictor.lstm(d)
        dur = torch.sigmoid(self.predictor.duration_proj(x)).sum(-1) / float(speed)
        pr = torch.round(dur.reshape(-1)).clamp(min=1).long()

        aln = torch.repeat_interleave(torch.eye(n_tok, device=device), pr, dim=1)
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        F0, N = self.predictor.F0Ntrain(en, sp)
        F0 = F0.unsqueeze(0) if F0.dim() == 1 else F0
        N = N.unsqueeze(0) if N.dim() == 1 else N
        waveform = self.decoder(t_en @ aln.unsqueeze(0), F0, N, sa)
        return {"waveform": waveform.reshape(-1), "duration": pr}
