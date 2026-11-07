import math

import torch
from torch import nn

from nnm.layers.embedding import merge_placeholders
from nnm.models.lfm3.language import Lfm3EmbeddingModel
from nnm.models.lfm3.omni import Lfm3OmniModel
from nnm.models.lfm3.vision import Lfm3VisionProjector, Lfm3VisionTower
from nnm.models.pretrained import PretrainedModel


def state_attention_mask(state_length, question_ids=None, valid=None, dtype=torch.float32):
    reference = question_ids if question_ids is not None else valid
    length = reference.shape[-1]
    device = reference.device
    queries = torch.arange(length, device=device)[None, :, None]
    keys = torch.arange(length, device=device)[None, None, :]
    states = state_length.to(device)[:, None, None]

    if question_ids is None:
        allowed = (keys <= queries) | ((queries < states) & (keys < states))
    else:
        allowed = (queries < states) | (keys < states) | (question_ids[:, :, None] == question_ids[:, None, :])
    if valid is not None:
        allowed = allowed & valid[:, None, :]

    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill(~allowed, float("-inf"))[:, None]


def position_logits(query, states, valid):
    dtype = torch.float64 if query.dtype == states.dtype == torch.float64 else torch.float32
    logits = torch.bmm(query.to(dtype), states.to(dtype).transpose(1, 2)) / math.sqrt(query.shape[-1])
    return logits.masked_fill(~valid[:, None], -torch.inf)


def pointer_probs(hidden, states, valid):
    return position_logits(hidden, states, valid).softmax(-1)


class D1VlModel(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)
        self.d1_attention = getattr(config, "d1_attention", None)
        self.d1_set_size = getattr(config, "d1_set_size", None)
        self.d1_codes = getattr(config, "d1_codes", None)
        self.vision_tower = Lfm3VisionTower(config.vision_config)
        self.multi_modal_projector = Lfm3VisionProjector(config)
        self.language_model = Lfm3EmbeddingModel(config.text_config)
        self.lm_head = (
            None
            if self.tie_word_embeddings
            else nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        )

    def get_image_features(self, pixel_values, grid_thw):
        hidden = self.vision_tower(pixel_values, grid_thw)
        features = self.multi_modal_projector(hidden)
        sizes = (grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        return torch.split(features, sizes)

    def merge_features(self, inputs_embeds, input_ids, token_id, features):
        return merge_placeholders(inputs_embeds, input_ids == token_id, features)

    def embed_inputs(self, input_ids, pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        for token_id, values, grid in (
            (self.image_token_id, pixel_values, image_grid_thw),
            (self.video_token_id, pixel_values_videos, video_grid_thw),
        ):
            if values is None or grid is None or token_id is None:
                continue
            inputs_embeds = self.merge_features(
                inputs_embeds, input_ids, token_id, torch.cat(self.get_image_features(values, grid))
            )
        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attn_mask=None,
    ):
        inputs_embeds = self.embed_inputs(input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        hidden = self.language_model(inputs_embeds=inputs_embeds, attn_mask=attn_mask)
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return torch.matmul(hidden, self.language_model.embed_tokens.weight.t())

    def clear_cache(self):
        if self.language_model.caches:
            for cache in self.language_model.caches:
                cache.clear()
        self.language_model.position_offset = 0


class D1OmniModel(Lfm3OmniModel):
    def __init__(self, config):
        super().__init__(config)
        self.d1_attention = getattr(config, "d1_attention", None)
        self.d1_set_size = getattr(config, "d1_set_size", None)
        self.d1_codes = getattr(config, "d1_codes", None)
