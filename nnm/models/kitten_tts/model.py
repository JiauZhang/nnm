import os

import numpy as np
import torch
from conippets.config import Config
from nnm.models.pretrained import PretrainedModel
from nnm.models.albert import AlbertEncoder

from .decoder import KittenDecoder
from .generator import KittenGenerator
from .predictor import Predictor
from .text import text_to_tokens
from .text_encoder import TextEncoder


class KittenTTS(PretrainedModel):
    def __init__(self, config, model_path=None, voices_path=None):
        super().__init__()
        self.config = config
        self.model_path = model_path

        self.bert = AlbertEncoder(config.albert)
        self.text_encoder = TextEncoder(config)
        self.predictor = Predictor(config)
        self.decoder = KittenDecoder(config)
        self.generator = KittenGenerator(config)

        if model_path is not None:
            self._load_weights(model_path)

        self._voices = None
        self._speed_priors = config.speed_priors.copy()
        self._voice_aliases = config.voice_aliases.copy()

        if voices_path:
            self.load_voices(voices_path)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        voices_path = os.path.join(pretrained_path, 'voices.npz')
        return super().from_pretrained(
            pretrained_path,
            voices_path=voices_path if os.path.exists(voices_path) else None,
        )

    def load_voices(self, voices_path):
        data = np.load(voices_path)
        self._voices = {k: data[k].copy() for k in data}
        data.close()

    def get_voice(self, name):
        if self._voices is None:
            raise RuntimeError('No voices loaded. Call load_voices() first.')
        key = self._voice_aliases.get(name, name)
        if key not in self._voices:
            available = list(self._voices.keys())
            raise KeyError(f'Voice "{name}" not found. Available: {available}')
        return self._voices[key]

    def get_speed(self, name, speed=None):
        if speed is not None:
            return speed
        key = self._voice_aliases.get(name, name)
        return self._speed_priors.get(key, self.config.speed)

    @torch.no_grad()
    def infer(self, input_ids, style=None, speed=1.0):
        device = next(self.parameters()).device
        if style is None:
            style = torch.randn(1, 256, device=device)
        elif style.dim() == 1:
            style = style.unsqueeze(0)

        bert_out = self.bert(input_ids)
        lstm_features, cnn_features = self.text_encoder(input_ids, style, bert_output=bert_out)
        te_out = lstm_features.permute(2, 0, 1)

        duration, expanded, shared, f0, n_amp = self.predictor(te_out, style, speed=speed)
        text_expanded = torch.repeat_interleave(cnn_features, duration, dim=2)

        stft_22ch = self.generator.m_source(f0)
        mel = self.decoder(text_expanded, f0, n_amp, style)
        waveform = self.generator(mel, style, harmonic_features=stft_22ch)

        return {
            'waveform': waveform,
            'duration': duration,
            'mel': mel,
            'stft_features': stft_22ch,
        }

    @torch.no_grad()
    def synthesize(
        self,
        text,
        voice='Bruno',
        speed=None,
        trim_end=0,
    ):
        if self._voices is None:
            raise RuntimeError("No voices loaded. Call load_voices('voices.npz') first.")

        voice_key = self._voice_aliases.get(voice, voice)
        if voice_key not in self._voices:
            voice_key = voice_key.lower()
        tokens = text_to_tokens(text)

        voice_embeddings = self._voices[voice_key]
        ref_id = min(len(text), voice_embeddings.shape[0] - 1)
        style_arr = voice_embeddings[ref_id:ref_id + 1]

        input_ids = torch.tensor([tokens], dtype=torch.long)
        style = torch.from_numpy(style_arr.copy()).float()
        speed_val = self.get_speed(voice, speed)

        result = self.infer(input_ids, style=style, speed=speed_val)

        waveform = result['waveform'].cpu().numpy()
        duration = result['duration'].cpu().numpy()

        if trim_end > 0:
            waveform = waveform[..., :-trim_end]

        return {
            'waveform': waveform.flatten(),
            'duration': duration,
            'sample_rate': self.config.sample_rate,
            'text': text,
            'voice': voice,
            'effective_speed': speed_val,
        }
