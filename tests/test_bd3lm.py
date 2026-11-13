import importlib.util
import os
import sys
import types

import pytest
import torch
from conippets.config import Config

from nnm.models.bd3lm import Bd3lm

REPOS = ["bd3lm-owt-block_size4", "bd3lm-owt-block_size8", "bd3lm-owt-block_size16"]


def _test_config(config, model_length=64):
    data = {k: config[k] for k in config.keys()}
    data["model_length"] = model_length
    return Config(**data)


def _forward(model, vocab_size, model_length, seed=0):
    torch.manual_seed(seed)
    indices = torch.randint(0, vocab_size, (1, 2 * model_length))
    sigma = torch.rand(1)
    with torch.no_grad():
        return model(indices, sigma, sample_mode=False)


@pytest.mark.parametrize("repo", REPOS)
def test_architecture(repo, weight_config):
    config = _test_config(weight_config(repo))
    model = Bd3lm(config).eval()
    logits = _forward(model, config.vocab_size, config.model_length)
    assert logits.shape == (1, config.model_length, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_pretrained_loads(model_path, weight_limit):
    if model_path is None:
        pytest.skip("--model-path required")
    weight_limit(model_path)
    model = Bd3lm.from_pretrained(model_path).eval()
    config = _test_config(model.config)
    model.backbone.gen_mask(config.model_length, config.block_size)
    model.backbone.n = config.model_length
    for block in model.backbone.blocks:
        block.n = config.model_length
    logits = _forward(model, config.vocab_size, config.model_length)
    assert torch.isfinite(logits).all()


def _load_hf(reference_path):
    hf_dir = os.path.join(reference_path, "models", "hf")
    pkg = types.ModuleType("bd3lm_hf")
    pkg.__path__ = [hf_dir]
    sys.modules["bd3lm_hf"] = pkg
    spec = importlib.util.spec_from_file_location(
        "bd3lm_hf.configuration_bd3lm", os.path.join(hf_dir, "configuration_bd3lm.py")
    )
    cfgmod = importlib.util.module_from_spec(spec)
    sys.modules["bd3lm_hf.configuration_bd3lm"] = cfgmod
    spec.loader.exec_module(cfgmod)
    spec = importlib.util.spec_from_file_location("bd3lm_hf.modeling_bd3lm", os.path.join(hf_dir, "modeling_bd3lm.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bd3lm_hf.modeling_bd3lm"] = mod
    spec.loader.exec_module(mod)
    return cfgmod.BD3LMConfig, mod.BD3LM


def test_matches_upstream(model_path, reference_path, weight_limit):
    if model_path is None or reference_path is None:
        pytest.skip("--model-path and --reference-path required")
    weight_limit(model_path)
    pytest.importorskip("einops")
    BD3LMConfig, HFBD3LM = _load_hf(reference_path)

    model = Bd3lm.from_pretrained(model_path).eval()
    config = _test_config(model.config)
    model.backbone.gen_mask(config.model_length, config.block_size)
    model.backbone.n = config.model_length
    for block in model.backbone.blocks:
        block.n = config.model_length

    hf_config = BD3LMConfig(
        hidden_dim=config.hidden_size,
        n_blocks=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        cond_dim=config.cond_dim,
        vocab_size=config.vocab_size,
        model_length=config.model_length,
        block_size=config.block_size,
        cross_attn=config.cross_attn,
        adaln=config.adaln,
        causal=config.causal,
        dropout=config.dropout,
        time_conditioning=config.time_conditioning,
        var_min=config.var_min,
        sampling_eps_min=config.sampling_eps_min,
        sampling_eps_max=config.sampling_eps_max,
        attn_backend="sdpa",
    )
    reference = HFBD3LM(hf_config).eval()
    reference.load_state_dict(model.state_dict(), strict=True)

    torch.manual_seed(0)
    indices = torch.randint(0, config.vocab_size, (1, 2 * config.model_length))
    sigma = torch.rand(1)
    with torch.no_grad():
        out = model(indices, sigma, sample_mode=False)
        ref = reference(indices, sigma, sample_mode=False).logits
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
