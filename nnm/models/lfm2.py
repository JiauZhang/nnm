import torch, re
from torch import nn
from nnm.layers.rope import QwenRoPE
from nnm.cache import KVCache, StateCache
from nnm.backends.sdpa import scaled_dot_product_attention


class Lfm2MLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size,
        intermediate_size,
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=256,
    ):
        super().__init__()
        if block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if block_ffn_dim_multiplier is not None:
                intermediate_size = int(block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = block_multiple_of * ((intermediate_size + block_multiple_of - 1) // block_multiple_of)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class Lfm2Attention(nn.Module):
    def __init__(self, *, config, position_encoder):
        super().__init__()
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.use_cache = getattr(config, "use_cache", False)
        self.position_encoder = position_encoder

        norm_eps = getattr(config, "norm_eps", 1e-6)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=True)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=True)

    def forward(self, hidden_states, attn_mask=None, cache=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape))
        key_states = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape))
        value_states = self.v_proj(hidden_states).view(*hidden_shape)

        use_cache = self.use_cache and cache is not None and not cache.is_empty()
        position = cache.kv_len if use_cache else None

        query_states = self.position_encoder(query_states, use_cache=use_cache, position=position)
        key_states = self.position_encoder(key_states, use_cache=use_cache, position=position)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_cache and cache is not None:
            cache_len = cache.kv_len
            seq_len = query_states.shape[-2]
            key_states, value_states = cache.update(key_states, value_states)
            if attn_mask is None and seq_len > 1:
                kv_len = key_states.shape[-2]
                mask = torch.full((seq_len, kv_len), float('-inf'), device=query_states.device, dtype=query_states.dtype)
                for i in range(seq_len):
                    mask[i, :cache_len + i + 1] = 0
                attn_mask = mask.unsqueeze(0).unsqueeze(0)
            is_causal = False
        else:
            is_causal = attn_mask is None

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=True,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = self.out_proj(attn_output)
        return output


class Lfm2ShortConv(nn.Module):
    def __init__(self, *, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.hidden_size = hidden_size
        self.L_cache = getattr(config, "conv_L_cache", 3)
        self.bias = getattr(config, "conv_bias", False)
        self.use_cache = getattr(config, "use_cache", False)

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.L_cache,
            groups=hidden_size,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=self.bias)

    def forward(self, hidden_states, cache=None, attn_mask=None):
        seqlen = hidden_states.shape[1]

        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        if self.use_cache and cache is not None and not cache.is_empty():
            conv_state = cache.update(Bx)
            conv_out = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1)
            if self.bias:
                conv_out += self.conv.bias
            conv_out = conv_out.unsqueeze(-1)
        else:
            conv_out = self.conv(Bx)[..., :seqlen]
            if self.use_cache and cache is not None:
                cache.update(Bx)

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, *, config, layer_idx, position_encoder, is_attention_layer, head_dim=None):
        super().__init__()
        self.is_attention_layer = is_attention_layer
        hidden_size = config.hidden_size
        norm_eps = getattr(config, "norm_eps", 1e-6)
        use_cache = getattr(config, "use_cache", False)

        if is_attention_layer:
            self.self_attn = Lfm2Attention(config=config, position_encoder=position_encoder)
        else:
            self.conv = Lfm2ShortConv(config=config)
        self.feed_forward = Lfm2MLP(
            hidden_size=hidden_size,
            intermediate_size=getattr(config, "intermediate_size", config.hidden_size * 4),
            block_auto_adjust_ff_dim=getattr(config, "block_auto_adjust_ff_dim", True),
            block_ffn_dim_multiplier=getattr(config, "block_ffn_dim_multiplier", 1.0),
            block_multiple_of=getattr(config, "block_multiple_of", 256),
        )
        self.operator_norm = nn.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True)

    def forward(self, hidden_states, attn_mask=None, cache=None):
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                attn_mask=attn_mask,
                cache=cache,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                cache=cache,
                attn_mask=attn_mask,
            )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

        return hidden_states


class Lfm2Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.use_cache = getattr(config, "use_cache", False)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.position_encoder = QwenRoPE(max_seq_len=config.max_position_embeddings, embed_dim=self.head_dim, base=rope_theta)

        full_attn_idxs = getattr(config, "full_attn_idxs", list(range(config.num_hidden_layers)))
        self.layer_types = ["full_attention" if i in full_attn_idxs else "conv" for i in range(config.num_hidden_layers)]

        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    position_encoder=self.position_encoder,
                    is_attention_layer=self.layer_types[layer_idx] == "full_attention",
                    head_dim=self.head_dim,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=getattr(config, "norm_eps", 1e-5), elementwise_affine=True)
        if self.use_cache:
            self.caches = []
            for layer_type in self.layer_types:
                if layer_type == "full_attention":
                    self.caches.append(KVCache())
                else:
                    self.caches.append(StateCache(state_size=getattr(config, "conv_L_cache", 3)))
        else:
            self.caches = None

    def forward(self, input_ids, attn_mask=None):
        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer, cache in zip(self.layers, self.caches or [None] * len(self.layers)):
            hidden_states = decoder_layer(
                hidden_states,
                attn_mask=attn_mask,
                cache=cache,
            )

        hidden_states = self.embedding_norm(hidden_states)
        return hidden_states


class Lfm2LM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.backbone = Lfm2Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(self, input_ids, attn_mask=None):
        output_embeds = self.backbone(input_ids, attn_mask=attn_mask)
        logits = self.lm_head(output_embeds)
        return logits

    def clear_cache(self):
        if self.backbone.caches:
            for cache in self.backbone.caches:
                cache.clear()

    def load_hf_state_dict(self, hf_state_dict):
        REPLACEMENT_PATTERNS = [
            (r"^model\.embed_tokens", "backbone.embed_tokens"),
            (r"^model\.embedding_norm", "backbone.embedding_norm"),
            (r"^model\.layers", "backbone.layers"),
        ]
        nnm_state_dict = {}
        for hf_key, tensor in hf_state_dict.items():
            nnm_key = hf_key
            for pattern, replacement in REPLACEMENT_PATTERNS:
                nnm_key = re.sub(pattern, replacement, nnm_key)
            nnm_state_dict[nnm_key] = tensor
        self.load_state_dict(nnm_state_dict)
