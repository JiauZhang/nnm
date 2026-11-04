import torch
from torch import nn

from nnm.layers.embedding import merge_placeholders
from nnm.models.pretrained import PretrainedModel
from nnm.models.minicpm_ocr.text import Qwen3_5Backbone
from nnm.models.minicpm_ocr.vision import MiniCPMOcrMerger, MiniCPMOcrVisionModel


class MiniCPMOcrModel(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.image_token_id = getattr(config, "image_token_id", None)
        self.downsample_mode = getattr(config, "downsample_mode", "4x")
        self.vision = MiniCPMOcrVisionModel(config.vision_config, insert_layer_id=config.insert_layer_id)
        self.backbone = Qwen3_5Backbone(config.text_config)
        self.merger = MiniCPMOcrMerger(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            merge_kernel_size=getattr(config, "merge_kernel_size", (2, 2)),
            merger_times=getattr(config, "merger_times", 1),
        )
        if self.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def get_image_features(self, pixel_values, target_sizes, downsample_mode=None):
        mode = downsample_mode or self.downsample_mode
        use_vit_merger = mode != "4x"
        hidden_states = self.vision(pixel_values, target_sizes, use_vit_merger=use_vit_merger)
        if use_vit_merger:
            target_sizes = target_sizes // 2
        return self.merger(hidden_states, target_sizes)

    def embed_inputs(self, input_ids, pixel_values=None, target_sizes=None, downsample_mode=None):
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        if pixel_values is not None and self.image_token_id is not None:
            image_features = torch.cat(self.get_image_features(pixel_values, target_sizes, downsample_mode), dim=0)
            inputs_embeds = merge_placeholders(inputs_embeds, input_ids == self.image_token_id, image_features)
        return inputs_embeds

    def forward(self, input_ids, pixel_values=None, target_sizes=None, position_ids=None, downsample_mode=None):
        inputs_embeds = self.embed_inputs(input_ids, pixel_values, target_sizes, downsample_mode)
        hidden_states = self.backbone(inputs_embeds=inputs_embeds, position_ids=position_ids)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return torch.matmul(hidden_states, self.backbone.embed_tokens.weight.t())

    def clear_cache(self):
        self.backbone.clear_cache()
