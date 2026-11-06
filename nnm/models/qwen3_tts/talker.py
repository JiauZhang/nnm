import torch
from functools import partial
from torch import nn

from nnm.cache import KVCache
from nnm.layers.attention import RotarySelfAttention
from nnm.layers.mlp import GatedMLP
from nnm.layers.norm import RMSNorm
from nnm.layers.rope import RoPE, rotary_inv_freq
from nnm.layers.transformer import TransformerBlock
from nnm.models.pretrained import PretrainedModel


def _make_layer(config):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    attention = RotarySelfAttention(
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
        qkv_bias=config.attention_bias,
        o_bias=config.attention_bias,
        qk_norm=True,
        norm_layer=partial(RMSNorm, eps=config.rms_norm_eps),
    )
    return TransformerBlock(
        config.hidden_size,
        attention,
        GatedMLP(config.hidden_size, config.intermediate_size),
        norm_layer=partial(RMSNorm, eps=config.rms_norm_eps),
    )


class TalkerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)
        self.layers = nn.ModuleList([_make_layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = RoPE(rotary_inv_freq(head_dim, config.rope_theta))
        self.caches = None
        self.position_offset = 0

    def prepare_cache(self):
        self.caches = [KVCache() for _ in range(len(self.layers))]
        self.position_offset = 0

    def clear_cache(self):
        if self.caches:
            for cache in self.caches:
                cache.clear()
        self.position_offset = 0

    def forward(self, inputs_embeds, attn_mask=None, use_cache=False):
        batch, seq_len = inputs_embeds.shape[:2]
        offset = self.position_offset
        positions = torch.arange(offset, offset + seq_len, device=inputs_embeds.device)
        cos, sin = self.rotary_emb(positions)
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                rotary_embeddings=(cos, sin),
                cache=self.caches[index] if use_cache else None,
                attn_mask=attn_mask,
            )
        if use_cache:
            self.position_offset += seq_len
        return self.norm(hidden)


class TextProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(config.text_hidden_size, config.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class CodePredictorModel(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([_make_layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = RoPE(rotary_inv_freq(head_dim, config.rope_theta))
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, hidden_size) for _ in range(config.num_code_groups - 1)]
        )
        self.caches = None
        self.position_offset = 0

    def prepare_cache(self):
        self.caches = [KVCache() for _ in range(len(self.layers))]
        self.position_offset = 0

    def clear_cache(self):
        if self.caches:
            for cache in self.caches:
                cache.clear()
        self.position_offset = 0

    def forward(self, inputs_embeds, use_cache=False):
        seq_len = inputs_embeds.shape[1]
        offset = self.position_offset
        positions = torch.arange(offset, offset + seq_len, device=inputs_embeds.device)
        cos, sin = self.rotary_emb(positions)
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                rotary_embeddings=(cos, sin),
                cache=self.caches[index] if use_cache else None,
            )
        if use_cache:
            self.position_offset += seq_len
        return self.norm(hidden)


class CodePredictor(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.model = CodePredictorModel(config, hidden_size)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

    def generate(self, past_hidden, token_embedding, use_cache=True, do_sample=False, top_k=50, top_p=1.0, temperature=0.9):
        if use_cache:
            self.model.prepare_cache()
        inputs_embeds = torch.cat([past_hidden, token_embedding], dim=1)
        hidden = self.model(inputs_embeds, use_cache=use_cache)
        logits = self.lm_head[0](hidden[:, -1])
        codes = [select_token(logits, do_sample, top_k, top_p, temperature)]

        for step in range(1, self.config.num_code_groups - 1):
            inputs_embeds = self.model.codec_embedding[step - 1](codes[-1].unsqueeze(-1))
            hidden = self.model(inputs_embeds, use_cache=use_cache)
            logits = self.lm_head[step](hidden[:, -1])
            codes.append(select_token(logits, do_sample, top_k, top_p, temperature))

        if use_cache:
            self.model.clear_cache()
        return torch.stack(codes, dim=-1)


def suppressed_logits(logits, config):
    if config.vocab_size <= 1024:
        return logits
    logits = logits.clone()
    eos = logits[..., config.codec_eos_token_id].clone()
    logits[..., config.vocab_size - 1024 :] = float("-inf")
    logits[..., config.codec_eos_token_id] = eos
    return logits


def select_token(logits, do_sample, top_k, top_p, temperature):
    if not do_sample:
        return logits.argmax(dim=-1)
    logits = logits.float() / max(temperature, 1e-5)
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, values)
    probabilities = logits.softmax(dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        mask = cumulative - sorted_probabilities > top_p
        sorted_probabilities = sorted_probabilities.masked_fill(mask, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter(-1, sorted_indices, sorted_probabilities)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


class Talker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TalkerModel(config)
        self.text_projection = TextProjection(config)
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.code_predictor = CodePredictor(config.code_predictor_config, config.hidden_size)

    def clear_cache(self):
        self.model.clear_cache()


class Qwen3TTS(PretrainedModel):
    module_name = "talker"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.talker = Talker(config.talker_config)

    def _text_embedding(self, input_ids):
        return self.talker.text_projection(self.talker.model.text_embedding(input_ids))

    def embed_prompt(self, input_ids, speaker, language, non_streaming_mode=True):
        config = self.config
        talker_config = config.talker_config
        device = input_ids.device

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._text_embedding(
            torch.tensor(
                [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
                device=device,
                dtype=input_ids.dtype,
            )
        ).chunk(3, dim=1)

        speaker_embed = self.talker.model.codec_embedding(
            torch.tensor(talker_config.spk_id[speaker.lower()], device=device, dtype=input_ids.dtype)
        )
        language = language.lower()
        language_id = None if language == "auto" else talker_config.codec_language_id[language]
        if language in ("chinese", "auto") and speaker and talker_config.spk_is_dialect[speaker.lower()] is not False:
            language_id = talker_config.codec_language_id[talker_config.spk_is_dialect[speaker.lower()]]

        if language_id is None:
            codec_prefill = [
                [
                    talker_config.codec_nothink_id,
                    talker_config.codec_think_bos_id,
                    talker_config.codec_think_eos_id,
                ]
            ]
        else:
            codec_prefill = [
                [
                    talker_config.codec_think_id,
                    talker_config.codec_think_bos_id,
                    language_id,
                    talker_config.codec_think_eos_id,
                ]
            ]
        codec_prefill_embed = self.talker.model.codec_embedding(
            torch.tensor(codec_prefill, device=device, dtype=input_ids.dtype)
        )
        codec_bos_embed = self.talker.model.codec_embedding(
            torch.tensor([[talker_config.codec_pad_id, talker_config.codec_bos_id]], device=device, dtype=input_ids.dtype)
        )
        codec_embed = torch.cat([codec_prefill_embed, speaker_embed.view(1, 1, -1), codec_bos_embed], dim=1)

        role_embed = self._text_embedding(input_ids[:, :3])
        prompt_embed = (
            torch.cat((tts_pad_embed.expand(-1, codec_embed.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_embed[:, :-1]
        )
        talker_embed = torch.cat((role_embed, prompt_embed), dim=1)

        talker_embed = torch.cat([talker_embed, self._text_embedding(input_ids[:, 3:4]) + codec_embed[:, -1:]], dim=1)
        if non_streaming_mode:
            talker_embed = talker_embed[:, :-1]
            text_embed = torch.cat(
                (self._text_embedding(input_ids[:, 3:-5]), tts_eos_embed), dim=1
            ) + self.talker.model.codec_embedding(
                torch.tensor(
                    [[talker_config.codec_pad_id] * (input_ids[:, 3:-5].shape[1] + 1)],
                    device=device,
                    dtype=input_ids.dtype,
                )
            )
            bos_embed = tts_pad_embed + self.talker.model.codec_embedding(
                torch.tensor([[talker_config.codec_bos_id]], device=device, dtype=input_ids.dtype)
            )
            talker_embed = torch.cat([talker_embed, text_embed, bos_embed], dim=1)
            trailing_text = tts_pad_embed
        else:
            trailing_text = torch.cat((self._text_embedding(input_ids[:, 4:-5]), tts_eos_embed), dim=1)

        return talker_embed, trailing_text, tts_pad_embed

    @torch.no_grad()
    def generate_codes(
        self,
        input_ids,
        speaker,
        language,
        max_new_tokens=4096,
        do_sample=False,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        subtalker_do_sample=False,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
        non_streaming_mode=True,
    ):
        talker = self.talker
        config = self.config.talker_config
        talker_embed, trailing_text, tts_pad_embed = self.embed_prompt(input_ids, speaker, language, non_streaming_mode)

        talker.model.prepare_cache()
        hidden = talker.model(talker_embed, use_cache=True)
        token = select_token(suppressed_logits(talker.codec_head(hidden[:, -1]), config), do_sample, top_k, top_p, temperature)
        past_hidden = hidden[:, -1:]
        frames = []

        for step in range(max_new_tokens):
            if token.item() == config.codec_eos_token_id:
                break
            token_embedding = talker.model.codec_embedding(token.unsqueeze(-1))
            group_codes = talker.code_predictor.generate(
                past_hidden,
                token_embedding,
                do_sample=subtalker_do_sample,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
                temperature=subtalker_temperature,
            )
            frame = torch.cat([token.unsqueeze(-1), group_codes], dim=-1)
            frames.append(frame)

            codec_embedding = token_embedding.clone()
            for index, code in enumerate(group_codes[0]):
                codec_embedding += talker.code_predictor.model.codec_embedding[index](code.view(1, 1))
            if step < trailing_text.shape[1]:
                codec_embedding = codec_embedding + trailing_text[:, step]
            else:
                codec_embedding = codec_embedding + tts_pad_embed

            hidden = talker.model(codec_embedding, use_cache=True)
            token = select_token(
                suppressed_logits(talker.codec_head(hidden[:, -1]), config), do_sample, top_k, top_p, temperature
            )
            past_hidden = hidden[:, -1:]

        talker.clear_cache()
        if not frames:
            return torch.zeros(0, self.config.talker_config.num_code_groups, dtype=torch.long)
        return torch.cat(frames, dim=0)

    def clear_cache(self):
        self.talker.clear_cache()
