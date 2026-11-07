import torch
from torch import nn

from nnm.layers.embedding import merge_placeholders
from nnm.models.lfm3.audio import AudioTower
from nnm.models.lfm3.language import Lfm3EmbeddingModel
from nnm.models.lfm3.vision import Lfm3VisionProjector, Lfm3VisionTower
from nnm.models.pretrained import PretrainedModel


class Lfm3OmniModel(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.audio_token_id = config.audio_token_id
        self.audio_out_token_id = config.audio_out_token_id
        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)
        self.language_model = Lfm3EmbeddingModel(config.text_config)
        self.audio = AudioTower(config)
        self.vision_tower = Lfm3VisionTower(config.vision_config)
        self.multi_modal_projector = Lfm3VisionProjector(config)
        self.lm_head = (
            None
            if self.tie_word_embeddings
            else nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        )

    def get_audio_features(self, input_features, feature_attention_mask):
        return self.audio.features(input_features, feature_attention_mask)

    def get_audio_out_embeds(self, audio_out_codes):
        return self.audio.code_embeds(audio_out_codes)

    def get_image_features(self, pixel_values, grid_thw):
        hidden = self.vision_tower(pixel_values, grid_thw)
        features = self.multi_modal_projector(hidden)
        sizes = (grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        return torch.split(features, sizes)

    def embed_inputs(
        self,
        input_ids,
        input_features=None,
        feature_attention_mask=None,
        audio_out_codes=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if input_features is not None:
            if feature_attention_mask is None:
                feature_attention_mask = torch.ones(
                    input_features.shape[0], input_features.shape[-1], dtype=torch.bool, device=input_features.device
                )
            features = self.get_audio_features(input_features, feature_attention_mask)
            inputs_embeds = self.merge_features(inputs_embeds, input_ids, self.audio_token_id, features)
        if audio_out_codes is not None:
            features = self.get_audio_out_embeds(audio_out_codes)
            inputs_embeds = self.merge_features(inputs_embeds, input_ids, self.audio_out_token_id, features)
        for token_id, values, grid in (
            (self.image_token_id, pixel_values, image_grid_thw),
            (self.video_token_id, pixel_values_videos, video_grid_thw),
        ):
            if values is None or grid is None or token_id is None:
                continue
            features = torch.cat(self.get_image_features(values, grid))
            inputs_embeds = self.merge_features(inputs_embeds, input_ids, token_id, features)
        return inputs_embeds

    def merge_features(self, inputs_embeds, input_ids, token_id, features):
        return merge_placeholders(inputs_embeds, input_ids == token_id, features)

    def forward(
        self,
        input_ids,
        input_features=None,
        feature_attention_mask=None,
        audio_out_codes=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attn_mask=None,
    ):
        inputs_embeds = self.embed_inputs(
            input_ids,
            input_features,
            feature_attention_mask,
            audio_out_codes,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )
        hidden = self.language_model(inputs_embeds=inputs_embeds, attn_mask=attn_mask)
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return torch.matmul(hidden, self.language_model.embed_tokens.weight.t())

    def audio_loss(self, hidden_states, input_ids, audio_labels):
        return self.audio.loss(hidden_states, input_ids, audio_labels)

    def clear_cache(self):
        if self.language_model.caches:
            for cache in self.language_model.caches:
                cache.clear()
        self.language_model.position_offset = 0
