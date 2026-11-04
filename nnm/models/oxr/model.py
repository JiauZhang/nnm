import torch

from nnm.models.pretrained import PretrainedModel

from .preprocess import NaViTPreprocessor
from .qwen3 import Qwen3ForCausalLM
from .visual import OXRVisualTokenizer, VisualVocabularyEmbedding
from .vit import Siglip2NaViT

VISUAL_ATOM_ID = -300
INDICATOR_IDS = (-301, -302, -303, -304)


class OXR(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.llm = Qwen3ForCausalLM(config.llm_config)
        self.visual_tokenizer = OXRVisualTokenizer(
            vit=Siglip2NaViT(config.vit_config),
            visual_vocab_size=config.visual_vocab_size,
            indicator_count=len(INDICATOR_IDS),
        )
        self.vte = VisualVocabularyEmbedding(config.visual_vocab_size, config.hidden_size)
        self.register_buffer(
            "indicator_token_indices",
            torch.arange(config.visual_vocab_size - len(INDICATOR_IDS), config.visual_vocab_size),
            persistent=False,
        )
        self.preprocessor = NaViTPreprocessor(
            patch_size=config.vit_config.patch_size,
            hidden_stride=config.vit_config.hidden_stride,
            temporal_patch_size=config.vit_config.temporal_patch_size,
        )

    def preprocess(self, images):
        return self.preprocessor(images)

    def visual_embed(self, visual_tokens):
        return self.vte(visual_tokens)

    def merge_multimodal(self, input_ids, pixel_values=None, grid_thws=None, visual_tokens=None):
        placeholder_mask = input_ids < 0
        multimodal_embeds = self.llm.backbone.embed_tokens(torch.masked_fill(input_ids, placeholder_mask, 0))

        if visual_tokens is None and pixel_values is not None:
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)

        if visual_tokens is not None:
            indicator_embeds = self.vte(self.indicator_token_indices).to(
                dtype=multimodal_embeds.dtype, device=multimodal_embeds.device
            )
            visual_embeds = self.vte(visual_tokens).to(dtype=multimodal_embeds.dtype, device=multimodal_embeds.device)
            for index, indicator_id in enumerate(INDICATOR_IDS):
                multimodal_embeds[input_ids == indicator_id] = indicator_embeds[index]
            multimodal_embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds

        return multimodal_embeds

    def forward(self, input_ids, attn_mask=None, pixel_values=None, grid_thws=None, visual_tokens=None):
        inputs_embeds = self.merge_multimodal(
            input_ids, pixel_values=pixel_values, grid_thws=grid_thws, visual_tokens=visual_tokens
        )
        return self.llm(inputs_embeds, attn_mask=attn_mask)

    def clear_cache(self):
        self.llm.clear_cache()
