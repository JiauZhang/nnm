import torch, pytest, random, os
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from transformers.models.qwen2 import configuration_qwen2 as cfg
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nnm.layers.rope import QwenRoPE
from nnm.models.qwen2 import (
    Qwen2Attention, Qwen2MLP, Qwen2DecoderLayer, Qwen2Backbone,
    Qwen2LM, make_causal_attn_mask,
)
from nnm.layers.norm import Qwen2RMSNorm
from conippets.config import Config

def init_linear_weight(src, dst, bias=True):
    assert dst.weight.shape == src.weight.shape
    dst.weight = src.weight
    if bias:
        assert dst.bias.shape == src.bias.shape
        dst.bias = src.bias

def init_linear_weight_kv(nnm_k_proj, nnm_v_proj, hf_k_proj, hf_v_proj):
    assert nnm_k_proj.weight.shape == hf_k_proj.weight.shape
    assert nnm_k_proj.bias.shape == hf_k_proj.bias.shape
    nnm_k_proj.weight = torch.nn.Parameter(hf_k_proj.weight.clone())
    nnm_k_proj.bias = torch.nn.Parameter(hf_k_proj.bias.clone())

    assert nnm_v_proj.weight.shape == hf_v_proj.weight.shape
    assert nnm_v_proj.bias.shape == hf_v_proj.bias.shape
    nnm_v_proj.weight = torch.nn.Parameter(hf_v_proj.weight.clone())
    nnm_v_proj.bias = torch.nn.Parameter(hf_v_proj.bias.clone())

@pytest.mark.parametrize(
    'batch, seq_len, max_seq_len, embed_dim, num_attn_heads, num_kv_heads, base', [
        (1, 234, 512, 32, 8, 4, 2333), (2, 345, 768, 64, 8, 8, 3456),
    ]
)
@torch.no_grad()
def test_qwen2_attn(batch, seq_len, max_seq_len, embed_dim, num_attn_heads, num_kv_heads, base):
    x = torch.randn(batch, seq_len, embed_dim)
    position_ids = torch.arange(seq_len)
    head_dim = embed_dim // num_attn_heads

    rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=head_dim, base=base)
    nnm_cos = rope.cos[position_ids, :]
    nnm_sin = rope.sin[position_ids, :]
    nnm_attn = Qwen2Attention(
        embed_dim=embed_dim, num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads, position_encoder=rope,
    )

    config = cfg.Qwen2Config(
        hidden_size=embed_dim, num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads,
        head_dim=head_dim, rope_theta=base, max_position_embeddings=max_seq_len,
    )
    config._attn_implementation = "sdpa"
    hf_attn = qwen2.Qwen2Attention(config, 0)

    init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj)
    nnm_q = nnm_attn.q_proj(x)
    hf_q = hf_attn.q_proj(x)
    assert nnm_q.shape == hf_q.shape
    torch.testing.assert_close(nnm_q, hf_q, atol=1e-5, rtol=1e-5)

    init_linear_weight(nnm_attn.o_proj, hf_attn.o_proj, bias=False)
    nnm_o = nnm_attn.o_proj(x)
    hf_o = hf_attn.o_proj(x)
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)

    init_linear_weight_kv(nnm_attn.k_proj, nnm_attn.v_proj, hf_attn.k_proj, hf_attn.v_proj)
    nnm_k, nnm_v = nnm_attn.k_proj(x), nnm_attn.v_proj(x)
    hf_k, hf_v = hf_attn.k_proj(x), hf_attn.v_proj(x)
    assert nnm_k.shape == hf_k.shape and nnm_v.shape == hf_v.shape
    torch.testing.assert_close(nnm_k, hf_k, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(nnm_v, hf_v, atol=1e-5, rtol=1e-5)

    hf_rope = qwen2.Qwen2RotaryEmbedding(config=config)
    position_ids = position_ids.unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)

    nnm_sin[:, :(head_dim//2)] = -nnm_sin[:, :(head_dim//2)]
    nnm_sin = nnm_sin.unsqueeze(0)
    nnm_cos = nnm_cos.unsqueeze(0)

    assert nnm_cos.shape == hf_cos.shape and nnm_sin.shape == hf_sin.shape
    torch.testing.assert_close(nnm_cos, hf_cos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(nnm_sin, hf_sin, atol=1e-5, rtol=1e-5)

    nnm_o = nnm_attn(x)
    hf_attn.eval()

    hf_o, hf_attn_weight = hf_attn(x, (hf_cos, hf_sin), None)
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize('batch, seq_len, embed_dim, eps', [(1, 123, 64, 1e-6), (2, 233, 128, 1e-8)])
@torch.no_grad()
def test_qwen2_rms_norm(batch, seq_len, embed_dim, eps):
    hf_norm = qwen2.Qwen2RMSNorm(embed_dim, eps=eps)
    nnm_norm = Qwen2RMSNorm(embed_dim, eps=eps)
    assert nnm_norm.weight.shape == hf_norm.weight.shape

    nnm_norm.weight = torch.nn.Parameter(torch.randn(*nnm_norm.weight.shape))
    x = torch.randn(batch, seq_len, embed_dim)
    hf_norm.weight = nnm_norm.weight
    nnm_o = nnm_norm(x)
    hf_o = hf_norm(x)
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)

def init_mlp_weight(src, dst):
    init_linear_weight(src.gate_proj, dst.gate_proj, bias=False)
    init_linear_weight(src.up_proj, dst.up_proj, bias=False)
    init_linear_weight(src.down_proj, dst.down_proj, bias=False)

@pytest.mark.parametrize('batch, seq_len, embed_dim, intermediate_size', [(1, 123, 64, 256), (2, 233, 128, 384)])
@torch.no_grad()
def test_qwen2_mlp(batch, seq_len, embed_dim, intermediate_size):
    config = cfg.Qwen2Config(hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu")
    hf_mlp = qwen2.Qwen2MLP(config)
    nnm_mlp = Qwen2MLP(embed_dim=embed_dim, intermediate_size=intermediate_size)

    init_mlp_weight(nnm_mlp, hf_mlp)
    x = torch.randn(batch, seq_len, embed_dim)
    nnm_o = nnm_mlp(x)
    hf_o = hf_mlp(x)
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)

def init_decoder_layer(nnm_decoder, hf_decoder):
    init_mlp_weight(nnm_decoder.mlp, hf_decoder.mlp)
    init_linear_weight(nnm_decoder.norm_1, hf_decoder.input_layernorm, bias=False)
    init_linear_weight(nnm_decoder.norm_2, hf_decoder.post_attention_layernorm, bias=False)
    nnm_attn, hf_attn = nnm_decoder.attn, hf_decoder.self_attn
    init_linear_weight(nnm_attn.q_proj, hf_attn.q_proj)
    init_linear_weight(nnm_attn.o_proj, hf_attn.o_proj, bias=False)
    init_linear_weight_kv(nnm_attn.k_proj, nnm_attn.v_proj, hf_attn.k_proj, hf_attn.v_proj)
    hf_decoder.eval()

@pytest.mark.parametrize(
    'batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps', [
        (1, 123, 512, 96, 256, 16, 4, 23432, 1e-6), (2, 233, 768, 128, 384, 32, 8, 10000, 1e-7),
    ]
)
@torch.no_grad()
def test_qwen2_decoder_layer(batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps):
    head_dim = embed_dim // num_attn_heads
    nnm_rope = QwenRoPE(max_seq_len=max_seq_len, embed_dim=head_dim, base=base)
    nnm_decoder = Qwen2DecoderLayer(
        position_encoder=nnm_rope, embed_dim=embed_dim, intermediate_size=intermediate_size,
        num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads, eps=eps,
    )
    config = cfg.Qwen2Config(
        hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu",
        num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads,
        rms_norm_eps=eps, head_dim=head_dim, rope_theta=base,
    )
    config._attn_implementation = "sdpa"
    hf_rope = qwen2.Qwen2RotaryEmbedding(config)
    hf_decoder = qwen2.Qwen2DecoderLayer(config, 0)

    init_decoder_layer(nnm_decoder, hf_decoder)

    x = torch.randn(batch, seq_len, embed_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    hf_cos, hf_sin = hf_rope(x, position_ids)
    attn_mask = None

    nnm_o = nnm_decoder(x, attn_mask=attn_mask)
    hf_o = hf_decoder(x, position_embeddings=(hf_cos, hf_sin), attention_mask=attn_mask)[0]
    if nnm_o.shape != hf_o.shape:
        nnm_o = nnm_o[0]
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


def init_qwen2_backbone(nnm_backbone, hf_backbone):
    hf_backbone.embed_tokens.weight = nnm_backbone.token_embeds.weight
    assert len(nnm_backbone.layers) == len(hf_backbone.layers)
    for nnm_decoder_layer, hf_decoder_layer in zip(nnm_backbone.layers, hf_backbone.layers):
        init_decoder_layer(nnm_decoder_layer, hf_decoder_layer)
    hf_backbone.norm.weight = nnm_backbone.norm.weight

def random_causal_mask(x):
    batch, seq_len = x.shape
    attn_mask = torch.ones_like(x, dtype=torch.float32)
    min_idx = max(1, (seq_len - 1) // 2)
    for b in range(batch):
        idx = random.randint(min_idx, seq_len-1)
        attn_mask[b, idx:] = 0
    return attn_mask

@pytest.mark.parametrize(
    ','.join([
        'batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps',
        'vocab_size, num_hidden_layers, sliding_window',
    ]), [
        (1, 123, 512, 96, 256, 16, 4, 23432, 1e-6, 1234, 6, 1024),
        (2, 233, 768, 128, 384, 32, 8, 10000, 1e-7, 2345, 9, 256),
    ]
)
@torch.no_grad()
def test_qwen2_backbone(
    batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps,
    vocab_size, num_hidden_layers, sliding_window,
):
    head_dim = embed_dim // num_attn_heads
    nnm_backbone = Qwen2Backbone(
        vocab_size=vocab_size, embed_dim=embed_dim, max_seq_len=max_seq_len, padding_idx=0, rms_norm_eps=eps,
        num_attn_heads=num_attn_heads, num_kv_heads=num_kv_heads, rope_base=base, intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers, sliding_window=sliding_window,
    )
    config = cfg.Qwen2Config(
        hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu",
        num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads, sliding_window=sliding_window,
        rms_norm_eps=eps, head_dim=head_dim, rope_theta=base, max_position_embeddings=max_seq_len,
        vocab_size=vocab_size, use_cache=False, num_hidden_layers=num_hidden_layers,
    )
    config._attn_implementation = "sdpa"
    hf_backbone = qwen2.Qwen2Model(config)

    init_qwen2_backbone(nnm_backbone, hf_backbone)

    x = torch.randint(0, vocab_size, (batch, seq_len))
    attn_mask = random_causal_mask(x)

    nnm_o = nnm_backbone(x, attn_mask=attn_mask)
    hf_o = hf_backbone(input_ids=x, attention_mask=attn_mask)[0]
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ','.join([
        'batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps',
        'vocab_size, num_hidden_layers, sliding_window',
    ]), [
        (1, 123, 512, 96, 256, 16, 4, 23233, 1e-6, 2134, 6, 1234),
        (2, 233, 768, 128, 384, 32, 8, 12345, 1e-7, 3211, 9, 256),
    ]
)
@torch.no_grad()
def test_qwen2_lm(
    batch, seq_len, max_seq_len, embed_dim, intermediate_size, num_attn_heads, num_kv_heads, base, eps,
    vocab_size, num_hidden_layers, sliding_window,
):
    head_dim = embed_dim // num_attn_heads
    nnm_config = Config(
        vocab_size=vocab_size, hidden_size=embed_dim, max_position_embeddings=max_seq_len, pad_token_id=0, rms_norm_eps=eps,
        num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads, rope_theta=base, intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers, sliding_window=sliding_window, use_kv_cache=False, tie_word_embeddings=False,
    )
    nnm_lm = Qwen2LM(nnm_config)
    hf_config = cfg.Qwen2Config(
        hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act="silu",
        num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads, sliding_window=sliding_window,
        rms_norm_eps=eps, head_dim=head_dim, rope_theta=base, max_position_embeddings=max_seq_len,
        vocab_size=vocab_size, use_cache=False, num_hidden_layers=num_hidden_layers,
    )
    hf_config._attn_implementation = "sdpa"
    hf_lm = qwen2.Qwen2ForCausalLM(hf_config)

    init_qwen2_backbone(nnm_lm.backbone, hf_lm.model)
    hf_lm.lm_head.weight = nnm_lm.lm_head.weight

    x = torch.randint(0, vocab_size, (batch, seq_len))
    attn_mask = random_causal_mask(x)

    nnm_o = nnm_lm(x, attn_mask=attn_mask)
    hf_o = hf_lm(input_ids=x, attention_mask=attn_mask)[0]
    assert nnm_o.shape == hf_o.shape
    torch.testing.assert_close(nnm_o, hf_o, atol=1e-5, rtol=1e-5)

    nnm_config = Config(
        vocab_size=vocab_size, hidden_size=embed_dim, max_position_embeddings=max_seq_len, pad_token_id=0, rms_norm_eps=eps,
        num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads, rope_theta=base, intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers, sliding_window=sliding_window, use_kv_cache=True, tie_word_embeddings=False,
    )
    nnm_lm = Qwen2LM(nnm_config)
    hf_config.use_cache = True
    hf_lm = qwen2.Qwen2ForCausalLM(hf_config)
    attn_mask = None
    for _ in range(seq_len):
        x = torch.randint(0, vocab_size, (batch, 1))
        nnm_o = nnm_lm(x, attn_mask=attn_mask)
        hf_o = hf_lm(input_ids=x, attention_mask=attn_mask)[0]
        assert nnm_o.shape == hf_o.shape


@torch.no_grad()
def test_qwen2_pretrained_model(model_path):
    if not model_path:
        pytest.skip("Model path not provided, use --model-path to specify")
    if not os.path.exists(model_path):
        pytest.skip(f"Model path not found: {model_path}")

    config = AutoConfig.from_pretrained(model_path)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    hf_model.eval()

    nnm_config = Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        max_position_embeddings=config.max_position_embeddings,
        pad_token_id=config.pad_token_id if config.pad_token_id is not None else 0,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_parameters['rope_theta'],
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.intermediate_size,
        sliding_window=config.sliding_window if config.sliding_window is not None else config.max_position_embeddings,
        num_hidden_layers=config.num_hidden_layers,
        use_kv_cache=False,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    nnm_model = Qwen2LM(nnm_config)
    nnm_model.load_hf_state_dict(hf_model.state_dict())
    nnm_model.eval()

    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids']

    hf_logits = hf_model(input_ids).logits
    nnm_logits = nnm_model(input_ids)

    assert hf_logits.shape == nnm_logits.shape

    torch.testing.assert_close(nnm_logits, hf_logits, atol=1e-4, rtol=1e-5)

    hf_pred = tokenizer.decode(hf_logits.argmax(dim=-1)[0])
    nnm_pred = tokenizer.decode(nnm_logits.argmax(dim=-1)[0])
    assert hf_pred == nnm_pred, f"Predictions don't match: hf='{hf_pred}', nnm='{nnm_pred}'"


@torch.no_grad()
def test_qwen2_pretrained_generation(model_path):
    if not model_path:
        pytest.skip("Model path not provided, use --model-path to specify")
    if not os.path.exists(model_path):
        pytest.skip(f"Model path not found: {model_path}")

    config = AutoConfig.from_pretrained(model_path)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    hf_model.eval()

    nnm_config = Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        max_position_embeddings=config.max_position_embeddings,
        pad_token_id=config.pad_token_id if config.pad_token_id is not None else 0,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_parameters['rope_theta'],
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.intermediate_size,
        sliding_window=config.sliding_window if config.sliding_window is not None else config.max_position_embeddings,
        num_hidden_layers=config.num_hidden_layers,
        use_kv_cache=False,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    nnm_model = Qwen2LM(nnm_config)
    nnm_model.load_hf_state_dict(hf_model.state_dict())
    nnm_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']

    hf_output = hf_model.generate(input_ids, max_new_tokens=5, do_sample=False)
    hf_text = tokenizer.decode(hf_output[0])

    nnm_input_ids = input_ids.clone()
    for _ in range(5):
        logits = nnm_model(nnm_input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        nnm_input_ids = torch.cat([nnm_input_ids, next_token], dim=1)
    nnm_text = tokenizer.decode(nnm_input_ids[0])

    assert hf_text == nnm_text, f"Generation doesn't match: hf='{hf_text}', nnm='{nnm_text}'"
