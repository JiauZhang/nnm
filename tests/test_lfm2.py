import torch, pytest, random, os
from torch import nn
from transformers.models.lfm2 import modeling_lfm2 as lfm2
from transformers.models.lfm2 import configuration_lfm2 as cfg
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nnm.layers.rope import QwenRoPE
from nnm.models.lfm2 import (
    Lfm2Attention,
    Lfm2MLP,
    Lfm2ShortConv,
    Lfm2DecoderLayer,
    Lfm2Backbone,
    Lfm2LM,
)
from conippets.config import Config


def init_linear_weight(src, dst, bias=True):
    assert dst.weight.shape == src.weight.shape
    dst.weight = src.weight
    if bias and hasattr(dst, "bias") and dst.bias is not None:
        assert dst.bias.shape == src.bias.shape
        dst.bias = src.bias


def init_mlp_weight(src, dst):
    init_linear_weight(src.w1, dst.w1, bias=False)
    init_linear_weight(src.w3, dst.w3, bias=False)
    init_linear_weight(src.w2, dst.w2, bias=False)


def init_decoder_layer(nnm_decoder, hf_decoder):
    init_mlp_weight(nnm_decoder.feed_forward, hf_decoder.feed_forward)
    nnm_decoder.operator_norm.weight = torch.nn.Parameter(hf_decoder.operator_norm.weight.clone())
    nnm_decoder.ffn_norm.weight = torch.nn.Parameter(hf_decoder.ffn_norm.weight.clone())

    if nnm_decoder.is_attention_layer:
        nnm_attn, hf_attn = nnm_decoder.self_attn, hf_decoder.self_attn
        init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj, bias=False)
        init_linear_weight(nnm_attn.k_proj, hf_attn.k_proj, bias=False)
        init_linear_weight(nnm_attn.v_proj, hf_attn.v_proj, bias=False)
        init_linear_weight(nnm_attn.out_proj, hf_attn.out_proj, bias=False)
        nnm_attn.q_layernorm.weight = torch.nn.Parameter(hf_attn.q_layernorm.weight.clone())
        nnm_attn.k_layernorm.weight = torch.nn.Parameter(hf_attn.k_layernorm.weight.clone())
    else:
        nnm_conv, hf_conv = nnm_decoder.conv, hf_decoder.conv
        nnm_conv.conv.weight = torch.nn.Parameter(hf_conv.conv.weight.clone())
        nnm_conv.in_proj.weight = torch.nn.Parameter(hf_conv.in_proj.weight.clone())
        nnm_conv.out_proj.weight = torch.nn.Parameter(hf_conv.out_proj.weight.clone())


def init_lfm2_backbone(nnm_backbone, hf_backbone):
    hf_backbone.embed_tokens.weight = nnm_backbone.embed_tokens.weight
    hf_backbone.embedding_norm.weight = nnm_backbone.embedding_norm.weight
    assert len(nnm_backbone.layers) == len(hf_backbone.layers)
    for nnm_decoder_layer, hf_decoder_layer in zip(nnm_backbone.layers, hf_backbone.layers):
        init_decoder_layer(nnm_decoder_layer, hf_decoder_layer)


def compute_rope_cos_sin(batch, seq_len, head_dim, base, position_ids):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].expand(batch, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


@pytest.mark.parametrize("batch, seq_len, hidden_size, eps", [(1, 123, 64, 1e-6), (2, 233, 128, 1e-5)])
@torch.no_grad()
def test_lfm2_rms_norm(batch, seq_len, hidden_size, eps):
    hf_norm = lfm2.Lfm2RMSNorm(hidden_size, eps=eps)
    nnm_norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=True)
    nnm_norm.weight = torch.nn.Parameter(torch.randn(*nnm_norm.weight.shape))
    hf_norm.weight = nnm_norm.weight

    x = torch.randn(batch, seq_len, hidden_size)
    torch.testing.assert_close(nnm_norm(x), hf_norm(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch, seq_len, hidden_size, intermediate_size", [(1, 123, 64, 256), (2, 233, 128, 384)])
@torch.no_grad()
def test_lfm2_mlp(batch, seq_len, hidden_size, intermediate_size):
    config = cfg.Lfm2Config(hidden_size=hidden_size, intermediate_size=intermediate_size, block_auto_adjust_ff_dim=False)
    hf_mlp = lfm2.Lfm2MLP(config)
    nnm_mlp = Lfm2MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, block_auto_adjust_ff_dim=False)
    init_mlp_weight(nnm_mlp, hf_mlp)

    x = torch.randn(batch, seq_len, hidden_size)
    torch.testing.assert_close(nnm_mlp(x), hf_mlp(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "batch, seq_len, max_seq_len, hidden_size, num_attention_heads, num_key_value_heads, base",
    [(1, 234, 512, 1024, 16, 8, 1000000.0), (2, 345, 768, 512, 8, 4, 10000.0)],
)
@torch.no_grad()
def test_lfm2_attn(batch, seq_len, max_seq_len, hidden_size, num_attention_heads, num_key_value_heads, base):
    head_dim = hidden_size // num_attention_heads
    x = torch.randn(batch, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    config = cfg.Lfm2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rope_theta=base,
        max_position_embeddings=max_seq_len,
        norm_eps=1e-6,
    )
    config._attn_implementation = "sdpa"

    nnm_config = Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        norm_eps=1e-6,
    )
    rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=head_dim, base=base)
    nnm_attn = Lfm2Attention(config=nnm_config, position_encoder=rope)
    hf_attn = lfm2.Lfm2Attention(config, layer_idx=0)

    init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj, bias=False)
    init_linear_weight(nnm_attn.k_proj, hf_attn.k_proj, bias=False)
    init_linear_weight(nnm_attn.v_proj, hf_attn.v_proj, bias=False)
    init_linear_weight(nnm_attn.out_proj, hf_attn.out_proj, bias=False)
    nnm_attn.q_layernorm.weight = torch.nn.Parameter(hf_attn.q_layernorm.weight.clone())
    nnm_attn.k_layernorm.weight = torch.nn.Parameter(hf_attn.k_layernorm.weight.clone())

    hf_rope = lfm2.Lfm2RotaryEmbedding(config=config)
    hf_cos, hf_sin = hf_rope(x, position_ids)

    nnm_o = nnm_attn(x)
    hf_o, _ = hf_attn(x, position_embeddings=(hf_cos, hf_sin), attention_mask=None)
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch, seq_len, hidden_size, conv_L_cache", [(1, 123, 64, 3), (2, 233, 128, 3)])
@torch.no_grad()
def test_lfm2_short_conv(batch, seq_len, hidden_size, conv_L_cache):
    config = cfg.Lfm2Config(hidden_size=hidden_size, conv_L_cache=conv_L_cache, conv_bias=False)
    hf_conv = lfm2.Lfm2ShortConv(config, layer_idx=0)
    nnm_conv = Lfm2ShortConv(config=Config(hidden_size=hidden_size, conv_L_cache=conv_L_cache, conv_bias=False))

    nnm_conv.conv.weight = torch.nn.Parameter(hf_conv.conv.weight.clone())
    nnm_conv.in_proj.weight = torch.nn.Parameter(hf_conv.in_proj.weight.clone())
    nnm_conv.out_proj.weight = torch.nn.Parameter(hf_conv.out_proj.weight.clone())

    x = torch.randn(batch, seq_len, hidden_size)
    torch.testing.assert_close(nnm_conv(x), hf_conv(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "batch, seq_len, max_seq_len, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, base, eps, is_attn",
    [(1, 123, 512, 1024, 2560, 16, 8, 1000000.0, 1e-5, True), (2, 233, 768, 512, 1280, 8, 4, 10000.0, 1e-5, False)],
)
@torch.no_grad()
def test_lfm2_decoder_layer(
    batch, seq_len, max_seq_len, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, base, eps, is_attn
):
    head_dim = hidden_size // num_attention_heads
    full_attn_idxs = [0] if is_attn else []

    config = cfg.Lfm2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rope_theta=base,
        max_position_embeddings=max_seq_len,
        norm_eps=eps,
        full_attn_idxs=full_attn_idxs,
        block_auto_adjust_ff_dim=False,
    )
    config._attn_implementation = "sdpa"

    nnm_config = Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        norm_eps=eps,
        full_attn_idxs=full_attn_idxs,
        block_auto_adjust_ff_dim=False,
    )
    rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=head_dim, base=base)
    nnm_decoder = Lfm2DecoderLayer(
        config=nnm_config, layer_idx=0, position_encoder=rope, is_attention_layer=is_attn, head_dim=head_dim
    )
    hf_decoder = lfm2.Lfm2DecoderLayer(config, layer_idx=0)
    init_decoder_layer(nnm_decoder, hf_decoder)

    x = torch.randn(batch, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    hf_rope = lfm2.Lfm2RotaryEmbedding(config=config)
    hf_cos, hf_sin = hf_rope(x, position_ids)

    nnm_o = nnm_decoder(x)
    hf_o = hf_decoder(x, position_embeddings=(hf_cos, hf_sin), attention_mask=None)
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "batch, seq_len, max_seq_len, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads",
    [(1, 123, 512, 1024, 2560, 16, 8), (2, 64, 512, 512, 1280, 8, 4)],
)
@torch.no_grad()
def test_lfm2_backbone(batch, seq_len, max_seq_len, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads):
    vocab_size, num_hidden_layers, full_attn_idxs = 1234, 8, [2, 5]
    head_dim = hidden_size // num_attention_heads

    nnm_config = Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_seq_len,
        pad_token_id=0,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rope_theta=1000000.0,
        norm_eps=1e-5,
        full_attn_idxs=full_attn_idxs,
        block_auto_adjust_ff_dim=False,
    )
    nnm_backbone = Lfm2Backbone(nnm_config)

    config = cfg.Lfm2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rope_theta=1000000.0,
        max_position_embeddings=max_seq_len,
        norm_eps=1e-5,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        full_attn_idxs=full_attn_idxs,
        block_auto_adjust_ff_dim=False,
    )
    config._attn_implementation = "sdpa"
    hf_backbone = lfm2.Lfm2Model(config)
    init_lfm2_backbone(nnm_backbone, hf_backbone)

    x = torch.randint(0, vocab_size, (batch, seq_len))
    torch.testing.assert_close(nnm_backbone(x), hf_backbone(input_ids=x)[0], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "batch, seq_len, max_seq_len, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, use_cache, start_with_multi_tokens",
    [
        (1, 10, 512, 1024, 2560, 16, 8, False, False),
        (1, 10, 512, 1024, 2560, 16, 8, True, False),
        (1, 10, 512, 1024, 2560, 16, 8, True, True),
        (2, 8, 512, 512, 1280, 8, 4, False, False),
        (2, 8, 512, 512, 1280, 8, 4, True, False),
        (2, 8, 512, 512, 1280, 8, 4, True, True),
    ],
)
@torch.no_grad()
def test_lfm2_lm(
    batch,
    seq_len,
    max_seq_len,
    hidden_size,
    intermediate_size,
    num_attention_heads,
    num_key_value_heads,
    use_cache,
    start_with_multi_tokens,
):
    vocab_size, num_hidden_layers, full_attn_idxs = 2134, 8, [2, 5]
    head_dim = hidden_size // num_attention_heads

    nnm_config = Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_seq_len,
        pad_token_id=0,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rope_theta=1000000.0,
        norm_eps=1e-5,
        full_attn_idxs=full_attn_idxs,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        use_cache=use_cache,
    )
    nnm_lm = Lfm2LM(nnm_config)

    config = cfg.Lfm2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rope_theta=1000000.0,
        max_position_embeddings=max_seq_len,
        norm_eps=1e-5,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        full_attn_idxs=full_attn_idxs,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        use_cache=use_cache,
    )
    config._attn_implementation = "sdpa"
    hf_lm = lfm2.Lfm2ForCausalLM(config)

    init_lfm2_backbone(nnm_lm.backbone, hf_lm.model)
    hf_lm.lm_head.weight = nnm_lm.lm_head.weight

    if not use_cache:
        x = torch.randint(0, vocab_size, (batch, seq_len))
        nnm_o = nnm_lm(x)
        hf_o = hf_lm(input_ids=x)[0]
        assert nnm_o.shape == hf_o.shape
        torch.testing.assert_close(nnm_o, hf_o, atol=1e-4, rtol=1e-4)
    else:
        attn_mask = None
        past_key_values = None

        if start_with_multi_tokens:
            start_seq_len = min(3, seq_len)
            x_start = torch.randint(0, vocab_size, (batch, start_seq_len))
            nnm_o_start = nnm_lm(x_start, attn_mask=attn_mask)
            hf_out_start = hf_lm(input_ids=x_start, attention_mask=attn_mask, past_key_values=past_key_values)
            hf_o_start = hf_out_start.logits
            past_key_values = hf_out_start.past_key_values
            assert nnm_o_start.shape == hf_o_start.shape
            torch.testing.assert_close(nnm_o_start, hf_o_start, atol=1e-5, rtol=1e-5)
            remaining_steps = seq_len - start_seq_len
        else:
            remaining_steps = seq_len

        for _ in range(remaining_steps):
            x = torch.randint(0, vocab_size, (batch, 1))
            nnm_o = nnm_lm(x, attn_mask=attn_mask)
            hf_out = hf_lm(input_ids=x, attention_mask=attn_mask, past_key_values=past_key_values)
            hf_o = hf_out.logits
            past_key_values = hf_out.past_key_values
            assert nnm_o.shape == hf_o.shape
            torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "prompts, use_cache",
    [
        (["A", " Tell me more."], False),
        (["A", " Tell me more."], True),
        (["Hello world", " What is it?"], False),
        (["Hello world", " What is it?"], True),
        (["Python is a programming", " How does it work?"], False),
        (["Python is a programming", " How does it work?"], True),
        (["What are you", " Can you explain?"], False),
        (["What are you", " Can you explain?"], True),
        (["Who is", " Tell me about them."], False),
        (["Who is", " Tell me about them."], True),
    ],
)
@torch.no_grad()
def test_lfm2_pretrained(model_path, prompts, use_cache):
    assert model_path is not None

    config = AutoConfig.from_pretrained(model_path)
    rope_theta = getattr(config, "rope_theta", 1000000.0)

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")
    hf_model.eval()

    nnm_config = Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        max_position_embeddings=config.max_position_embeddings,
        pad_token_id=config.pad_token_id or 0,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rope_theta=rope_theta,
        norm_eps=config.norm_eps,
        full_attn_idxs=config.full_attn_idxs,
        conv_L_cache=config.conv_L_cache,
        conv_bias=config.conv_bias,
        block_auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
        block_ffn_dim_multiplier=config.block_ffn_dim_multiplier,
        block_multiple_of=config.block_multiple_of,
        tie_word_embeddings=config.tie_word_embeddings,
        use_cache=use_cache,
    )
    nnm_model = Lfm2LM(nnm_config)
    nnm_model.load_hf_state_dict(hf_model.state_dict())
    nnm_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer(prompts[0], return_tensors="pt")["input_ids"]

    hf_logits = hf_model(input_ids).logits
    nnm_logits = nnm_model(input_ids)
    torch.testing.assert_close(nnm_logits, hf_logits, atol=1e-4, rtol=1e-4)
    assert tokenizer.decode(hf_logits.argmax(dim=-1)[0]) == tokenizer.decode(nnm_logits.argmax(dim=-1)[0])

    max_new_tokens = 10

    hf_input_ids = tokenizer(prompts[0], return_tensors="pt")["input_ids"]
    nnm_input_ids = hf_input_ids.clone()
    hf_past_key_values = None

    for turn_idx, prompt in enumerate(prompts):
        if turn_idx > 0:
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            hf_input_ids = torch.cat([hf_input_ids, input_ids], dim=1)
            nnm_input_ids = torch.cat([nnm_input_ids, input_ids], dim=1)
        else:
            input_ids = hf_input_ids

        if use_cache:
            hf_out = hf_model(input_ids, past_key_values=hf_past_key_values, use_cache=True)
            hf_past_key_values = hf_out.past_key_values
            hf_logits = hf_out.logits
            logits = nnm_model(input_ids)
        else:
            hf_out = hf_model(hf_input_ids, use_cache=False)
            hf_logits = hf_out.logits
            logits = nnm_model(nnm_input_ids)

        next_token = hf_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        hf_input_ids = torch.cat([hf_input_ids, next_token], dim=1)
        nnm_input_ids = torch.cat([nnm_input_ids, next_token], dim=1)

        for _ in range(1, max_new_tokens):
            if use_cache:
                hf_out = hf_model(hf_input_ids[:, -1:], past_key_values=hf_past_key_values, use_cache=True)
                hf_past_key_values = hf_out.past_key_values
                hf_logits = hf_out.logits
                logits = nnm_model(nnm_input_ids[:, -1:])
            else:
                hf_out = hf_model(hf_input_ids, use_cache=False)
                hf_logits = hf_out.logits
                logits = nnm_model(nnm_input_ids)

            next_token = hf_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            hf_input_ids = torch.cat([hf_input_ids, next_token], dim=1)
            nnm_input_ids = torch.cat([nnm_input_ids, next_token], dim=1)

    hf_text = tokenizer.decode(hf_input_ids[0])
    nnm_text = tokenizer.decode(nnm_input_ids[0])
    assert hf_text == nnm_text, f"Generation doesn't match: hf='{hf_text}', nnm='{nnm_text}'"
