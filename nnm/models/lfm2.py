import torch, re
from torch import nn
from nnm.layers.rope import QwenRoPE


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Lfm2Attention(nn.Module):
    def __init__(self, *, config):
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
        self.use_kv_cache = getattr(config, "use_kv_cache", False)

        norm_eps = getattr(config, "norm_eps", 1e-6)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.q_layernorm = Lfm2RMSNorm(self.head_dim, eps=norm_eps)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim, eps=norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attn_mask=None, kv_cache=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        key_states = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_kv_cache and kv_cache is not None:
            kv_cache[0].append(key_states)
            kv_cache[1].append(value_states)
            key_states = torch.cat(kv_cache[0], dim=-2)
            value_states = torch.cat(kv_cache[1], dim=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        is_causal = attn_mask is None
        attn_output = torch.nn.functional.scaled_dot_product_attention(
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
        self.use_kv_cache = getattr(config, "use_kv_cache", False)

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

    def forward(self, hidden_states, kv_cache=None, attn_mask=None):
        seqlen = hidden_states.shape[1]

        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        if self.use_kv_cache and kv_cache is not None and len(kv_cache) > 0:
            conv_state = kv_cache[0]
            if conv_state is not None and conv_state.shape[-1] > 0:
                conv_state = torch.cat([conv_state, Bx], dim=-1)
                conv_state = conv_state[..., -self.L_cache :]
            else:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
            kv_cache[0] = conv_state
            conv_out = torch.sum(conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
            if self.bias:
                conv_out += self.conv.bias
            conv_out = conv_out.unsqueeze(-1)
        else:
            if self.use_kv_cache and kv_cache is not None:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                kv_cache[0] = conv_state
            conv_out = self.conv(Bx)[..., :seqlen]

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
        use_kv_cache = getattr(config, "use_kv_cache", False)

        if is_attention_layer:
            self.self_attn = Lfm2Attention(config=config)
        else:
            self.conv = Lfm2ShortConv(config=config)
        self.feed_forward = Lfm2MLP(
            hidden_size=hidden_size,
            intermediate_size=getattr(config, "intermediate_size", config.hidden_size * 4),
            block_auto_adjust_ff_dim=getattr(config, "block_auto_adjust_ff_dim", True),
            block_ffn_dim_multiplier=getattr(config, "block_ffn_dim_multiplier", 1.0),
            block_multiple_of=getattr(config, "block_multiple_of", 256),
        )
        self.operator_norm = Lfm2RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = Lfm2RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attn_mask=None, kv_cache=None):
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                kv_cache=kv_cache,
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
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=getattr(config, "norm_eps", 1e-5))
        self.kv_cache = [
            ([torch.empty(0)] if self.layer_types[i] == "conv" else ([], [])) for i in range(config.num_hidden_layers)
        ]

    def forward(self, input_ids, attn_mask=None):
        input_embeds = self.embed_tokens(input_ids)
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        # Compute cos/sin like transformers RoPE
        # inv_freq shape: [head_dim // 2]
        inv_freq = 1.0 / (
            self.position_encoder.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=input_ids.device) / self.head_dim)
        )
        # inv_freq_expanded: [batch, head_dim // 2, 1]
        inv_freq_expanded = inv_freq[None, :, None].expand(batch, -1, 1)
        # position_ids_expanded: [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        # freqs: [batch, seq_len, head_dim // 2]
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # emb: [batch, seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        hidden_states = input_embeds

        for decoder_layer, kv_cache in zip(self.layers, self.kv_cache):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=(cos, sin),
                attn_mask=attn_mask,
                kv_cache=kv_cache,
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

    def load_hf_state_dict(self, hf_state_dict):
        REPLACEMENT_PATTERNS = [
            (r"^model\.embed_tokens", "backbone.embed_tokens"),
            (r"^model\.embedding_norm", "backbone.embedding_norm"),
            (r"^model\.layers", "backbone.layers"),
            (r"^lm_head", "lm_head"),
        ]
        nnm_state_dict = {}
        for hf_key, tensor in hf_state_dict.items():
            nnm_key = hf_key
            for pattern, replacement in REPLACEMENT_PATTERNS:
                nnm_key = re.sub(pattern, replacement, nnm_key)
            nnm_state_dict[nnm_key] = tensor
        self.load_state_dict(nnm_state_dict)
