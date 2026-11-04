import torch
from torch import nn

from nnm.layers.embedding import merge_placeholders
from nnm.models.pretrained import PretrainedModel
from nnm.models.qwen3_asr.audio import Qwen3ASRAudioEncoder
from nnm.models.qwen3_asr.text import Qwen3ASRTextModel


class Qwen3ASRThinker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_tower = Qwen3ASRAudioEncoder(config.audio_config)
        self.model = Qwen3ASRTextModel(config.text_config)
        classify_num = getattr(config, "classify_num", None)
        self.lm_head = nn.Linear(config.text_config.hidden_size, classify_num or config.text_config.vocab_size, bias=False)

    def get_audio_features(self, input_features, feature_attention_mask):
        if feature_attention_mask is None:
            raise ValueError("feature_attention_mask is required")
        feature_lens = feature_attention_mask.sum(-1)
        outputs = []
        for feature, length in zip(input_features, feature_lens):
            outputs.append(self.audio_tower(feature[:, :length], length.unsqueeze(0)))
        return torch.cat(outputs, dim=0)

    def embed_inputs(self, input_ids, input_features=None, feature_attention_mask=None):
        inputs_embeds = self.model.embed_tokens(input_ids)
        if input_features is not None:
            audio_features = self.get_audio_features(input_features, feature_attention_mask)
            inputs_embeds = merge_placeholders(inputs_embeds, input_ids == self.config.audio_token_id, audio_features)
        return inputs_embeds

    def forward(self, input_ids, input_features=None, feature_attention_mask=None, attn_mask=None, use_cache=False):
        inputs_embeds = self.embed_inputs(input_ids, input_features, feature_attention_mask)
        hidden = self.model(inputs_embeds, attn_mask=attn_mask, use_cache=use_cache)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self, input_ids, input_features=None, feature_attention_mask=None, max_new_tokens=256, eos_token_ids=(151645, 151643)
    ):
        inputs_embeds = self.embed_inputs(input_ids, input_features, feature_attention_mask)
        self.model.prepare_cache()
        hidden = self.model(inputs_embeds, use_cache=True)
        token = self.lm_head(hidden[:, -1]).argmax(-1)
        tokens = []
        for _ in range(max_new_tokens):
            if token.item() in eos_token_ids:
                break
            tokens.append(token.item())
            hidden = self.model(self.model.embed_tokens(token[:, None]), use_cache=True)
            token = self.lm_head(hidden[:, -1]).argmax(-1)
        self.model.clear_cache()
        return tokens


class Qwen3ASR(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.thinker = Qwen3ASRThinker(config.thinker_config)

    def forward(self, input_ids, input_features=None, feature_attention_mask=None, attn_mask=None):
        return self.thinker(input_ids, input_features, feature_attention_mask, attn_mask)

    def generate(self, input_ids, input_features=None, feature_attention_mask=None, max_new_tokens=256):
        return self.thinker.generate(input_ids, input_features, feature_attention_mask, max_new_tokens)


class Qwen3ForcedAligner(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.thinker = Qwen3ASRThinker(config.thinker_config)
        self.timestamp_token_id = int(config.timestamp_token_id)
        self.timestamp_segment_time = float(config.timestamp_segment_time)

    def forward(self, input_ids, input_features=None, feature_attention_mask=None, attn_mask=None):
        return self.thinker(input_ids, input_features, feature_attention_mask, attn_mask)

    @torch.no_grad()
    def align(self, input_ids, input_features, feature_attention_mask=None):
        logits = self.thinker(input_ids, input_features, feature_attention_mask)[0]
        timestamps = logits[input_ids[0] == self.timestamp_token_id].argmax(-1)
        return (timestamps * self.timestamp_segment_time / 1000.0).tolist()
