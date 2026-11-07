import torch
from torch import nn

from nnm.layers.embedding import merge_placeholders
from nnm.models.lfm3.language import Lfm3EmbeddingModel
from nnm.models.lfm3.vision import Lfm3VlMultiModalProjector, Lfm3VlVisionModel
from nnm.models.pretrained import PretrainedModel


class Lfm3VlModel(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.image_token_id = config.image_token_id
        self.vision_tower = Lfm3VlVisionModel(config.vision_config)
        self.multi_modal_projector = Lfm3VlMultiModalProjector(config)
        self.backbone = Lfm3EmbeddingModel(config.text_config)
        self.lm_head = (
            None
            if self.tie_word_embeddings
            else nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        )

    def get_image_features(self, pixel_values, image_grid_thw):
        hidden_states = self.vision_tower(pixel_values, image_grid_thw)
        features = self.multi_modal_projector(hidden_states)
        sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        return torch.split(features, sizes)

    def embed_inputs(self, input_ids, pixel_values=None, image_grid_thw=None):
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        if pixel_values is not None:
            image_features = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0)
            inputs_embeds = merge_placeholders(inputs_embeds, input_ids == self.image_token_id, image_features)
        return inputs_embeds

    def forward(self, input_ids, pixel_values=None, image_grid_thw=None, attn_mask=None):
        inputs_embeds = self.embed_inputs(input_ids, pixel_values, image_grid_thw)
        hidden_states = self.backbone(inputs_embeds=inputs_embeds, attn_mask=attn_mask)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return torch.matmul(hidden_states, self.backbone.embed_tokens.weight.t())

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()
        self.backbone.position_offset = 0
