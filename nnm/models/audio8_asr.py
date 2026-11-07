import torch
import torch.nn.functional as F
from torch import nn

from nnm.layers.mlp import MLP
from nnm.models.pretrained import PretrainedModel
from nnm.models.qwen2 import Qwen2Model
from nnm.models.qwen3_asr.audio import Qwen3ASRAudioEncoder


class Audio8MLPTowerBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, "gelu", bias=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        return hidden + self.dropout(self.mlp(self.norm(hidden)))


class Audio8MLPTower(nn.Module):
    def __init__(self, hidden_size, num_layers, intermediate_size, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([Audio8MLPTowerBlock(hidden_size, intermediate_size, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.final_norm(hidden)


class Audio8ASR(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_token_id = int(config.get("audio_token_id"))
        audio_config = config.get("qwen3_asr_audio_config")
        audio_dim = int(audio_config.output_dim)

        self.language_model = nn.Module()
        self.language_model.model = Qwen2Model(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            max_seq_len=config.max_position_embeddings,
            padding_idx=config.get("pad_token_id", 0),
            num_hidden_layers=config.num_hidden_layers,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=config.rope_theta,
            num_attn_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            sliding_window=config.get("sliding_window"),
            use_cache=config.get("use_cache", False),
        )
        self.audio_encoder = Qwen3ASRAudioEncoder(audio_config)
        intermediate_size = config.get("qwen3_asr_mlp_tower_hidden_size", 0) or audio_dim * 4
        self.audio_mlp_tower = Audio8MLPTower(
            audio_dim,
            config.get("qwen3_asr_mlp_tower_layers", 4),
            intermediate_size,
            config.get("qwen3_asr_mlp_tower_dropout", 0.0),
        )
        self.audio_projector = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, config.hidden_size))

    def get_input_embeddings(self):
        return self.language_model.model.token_embeds

    def _project_audio_row(self, input_features, token_count, dtype, feature_length=None):
        if token_count <= 0:
            return input_features.new_zeros((0, self.get_input_embeddings().embedding_dim), dtype=dtype)
        if input_features.dim() == 3 and input_features.size(0) == 1:
            input_features = input_features.squeeze(0)
        expected_mels = int(self.audio_encoder.num_mel_bins)
        if input_features.size(0) != expected_mels and input_features.size(1) == expected_mels:
            input_features = input_features.transpose(0, 1)
        if input_features.size(0) != expected_mels:
            raise ValueError(f"Audio feature bins mismatch: expected {expected_mels}, got {input_features.size(0)}")

        feature_length = int(input_features.size(1)) if feature_length is None else int(feature_length)
        feature_length = max(1, min(feature_length, int(input_features.size(1))))
        input_features = input_features[:, :feature_length]

        encoder_param = next(self.audio_encoder.parameters())
        encoded = self.audio_encoder(
            input_features.to(device=encoder_param.device, dtype=encoder_param.dtype),
            feature_lens=torch.tensor([feature_length], dtype=torch.long, device=encoder_param.device),
        )
        tower_param = next(self.audio_mlp_tower.parameters())
        hidden = self.audio_mlp_tower(encoded.to(device=tower_param.device, dtype=tower_param.dtype))
        if int(hidden.size(0)) != token_count:
            hidden = (
                F.adaptive_avg_pool1d(
                    hidden.transpose(0, 1).float().unsqueeze(0),
                    output_size=token_count,
                )
                .squeeze(0)
                .transpose(0, 1)
                .to(dtype=tower_param.dtype)
            )
        projector_param = next(self.audio_projector.parameters())
        return self.audio_projector(hidden.to(device=projector_param.device, dtype=projector_param.dtype)).to(dtype=dtype)

    def _inject_audio_embeddings(self, input_ids, input_features, feature_lens=None):
        input_embeddings = self.get_input_embeddings()(input_ids).clone()
        if input_features is None:
            return input_embeddings
        input_features = input_features.to(device=input_embeddings.device)
        if input_features.dim() == 4 and input_features.size(1) == 1:
            input_features = input_features.squeeze(1)
        audio_mask = input_ids.eq(self.audio_token_id)
        for batch_index in range(int(input_ids.size(0))):
            positions = torch.nonzero(audio_mask[batch_index], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            feature_length = None
            if feature_lens is not None and batch_index < int(feature_lens.numel()):
                feature_length = int(feature_lens[batch_index].item())
            projected = self._project_audio_row(
                input_features[batch_index],
                int(positions.numel()),
                input_embeddings.dtype,
                feature_length=feature_length,
            )
            input_embeddings[batch_index, positions, :] = projected.to(device=input_embeddings.device)
        return input_embeddings

    def forward(
        self,
        input_ids,
        input_features=None,
        feature_lens=None,
        audios=None,
        audio_feature_lengths=None,
        attention_mask=None,
    ):
        if input_features is None:
            input_features = audios
        if feature_lens is None:
            feature_lens = audio_feature_lengths
        if feature_lens is not None and not torch.is_tensor(feature_lens):
            feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=input_ids.device)
        inputs_embeds = self._inject_audio_embeddings(input_ids, input_features, feature_lens)
        hidden = self.language_model.model(inputs_embeds=inputs_embeds, attn_mask=attention_mask)
        return torch.matmul(hidden, self.get_input_embeddings().weight.t())
