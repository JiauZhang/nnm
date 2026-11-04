import json
import os

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from nnm.models.timestral import ROLE_KNOWN_COVARIATE, ROLE_PAST_COVARIATE, ROLE_TARGET, Timestral


@pytest.fixture(scope="module")
def model(model_path):
    assert model_path is not None, "--model-path required"
    return Timestral.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def reference(model, model_path):
    reference_module = pytest.importorskip("timestral")
    manifest = json.loads(open(os.path.join(model_path, "nnm_weights.json")).read())
    source = manifest["sources"][0]
    directory = snapshot_download(source["repo"], allow_patterns=[source["pattern"], "config.json"])
    config = reference_module.TimestralConfig.load(os.path.join(directory, "config.json"))
    reference_model = reference_module.TimestralModel(config).eval()
    reference_model.load_state_dict(load_file(os.path.join(directory, source["pattern"]), device="cpu"), strict=True)
    return reference_model


def make_context(batch, variates, length, missing=False):
    generator = torch.Generator().manual_seed(0)
    context = torch.randn(batch, variates, length, generator=generator).cumsum(-1)
    observed = torch.ones(batch, variates, length, dtype=torch.bool)
    if missing and batch > 1:
        observed[1, 0, length // 2 :] = False
    return context, observed


def pack(context, observed, extra_values=None, extra_observed=None):
    batch, variates, length = context.shape
    if extra_values is not None:
        context = torch.cat([context, extra_values], dim=1)
        observed = torch.cat([observed, extra_observed], dim=1)
    return context.reshape(batch * context.shape[1], length), observed.reshape(batch * context.shape[1], length)


class TestTimestral:
    def test_config_drives_the_architecture(self, model):
        assert model.config.patch_size == 32
        assert model.config.d_model == 192
        assert model.config.num_heads == 3
        assert sum(parameter.numel() for parameter in model.parameters()) == 5208960

    @pytest.mark.parametrize("batch, variates, length, horizon", [(1, 1, 128, 2), (2, 1, 256, 4), (2, 2, 192, 3)])
    def test_output_shapes(self, model, batch, variates, length, horizon):
        context, observed = make_context(batch, variates, length)
        inputs, mask = pack(context, observed)
        group_ids = torch.arange(batch).repeat_interleave(variates)
        with torch.no_grad():
            output = model(inputs, mask, group_ids, horizon)

        assert output.shape == (batch * variates, horizon * model.config.patch_size, len(model.config.quantile_knots))

    @pytest.mark.parametrize("batch, variates, length, horizon", [(1, 1, 128, 2), (2, 2, 192, 3)])
    def test_quantiles_are_monotone(self, model, batch, variates, length, horizon):
        context, observed = make_context(batch, variates, length, missing=True)
        inputs, mask = pack(context, observed)
        group_ids = torch.arange(batch).repeat_interleave(variates)
        with torch.no_grad():
            output = model(inputs, mask, group_ids, horizon)

        assert torch.all(output[..., 1:] >= output[..., :-1])
        assert torch.isfinite(output).all()

    def test_context_only_matches_the_reference(self, model, reference):
        context, observed = make_context(2, 2, 128, missing=True)
        inputs, mask = pack(context, observed)
        group_ids = torch.arange(2).repeat_interleave(2)
        with torch.no_grad():
            expected = reference.predict_block(inputs, mask, group_ids, 4)
            actual = model(inputs, mask, group_ids, 4)

        assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-4)

    def test_covariates_match_the_reference(self, model, reference):
        batch, horizon = 2, 3
        context, observed = make_context(batch, 2, 128)
        past_values = torch.randn(batch, 1, 128, generator=torch.Generator().manual_seed(1))
        past_observed = torch.ones(batch, 1, 128, dtype=torch.bool)
        known_context = torch.randn(batch, 1, 128, generator=torch.Generator().manual_seed(2))
        known_context_observed = torch.ones(batch, 1, 128, dtype=torch.bool)
        inputs, mask = pack(
            context,
            observed,
            torch.cat([past_values, known_context], dim=1),
            torch.cat([past_observed, known_context_observed], dim=1),
        )

        known_values = torch.randn(batch, 1, horizon * model.config.patch_size)
        known_observed = torch.ones(batch, 1, horizon * model.config.patch_size, dtype=torch.bool)
        rows = batch * 4
        known_future = torch.zeros(rows, horizon * model.config.patch_size)
        known_future_mask = torch.zeros(rows, horizon * model.config.patch_size, dtype=torch.bool)
        known_future.view(batch, 4, -1)[:, 3] = known_values.squeeze(1)
        known_future_mask.view(batch, 4, -1)[:, 3] = known_observed.squeeze(1)
        group_ids = torch.arange(batch).repeat_interleave(4)
        role_ids = torch.tensor([ROLE_TARGET, ROLE_TARGET, ROLE_PAST_COVARIATE, ROLE_KNOWN_COVARIATE]).repeat(batch)

        with torch.no_grad():
            expected = reference.predict_block(
                inputs,
                mask,
                group_ids,
                horizon,
                role_ids=role_ids,
                known_future=known_future,
                known_future_mask=known_future_mask,
            )
            actual = model(
                inputs,
                mask,
                group_ids,
                horizon,
                role_ids=role_ids,
                known_future=known_future,
                known_future_mask=known_future_mask,
            )

        assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-4)
