import os

import numpy as np
import pytest
import torch

from nnm.models.indra_ocr import (
    IndraOCRDetection,
    IndraOCROrder,
    IndraOCROrientation,
    IndraOCRRecognition,
    IndraOCRScript,
)
from nnm.models.indra_ocr.lm import NgramLM, _pack, beam_search


@pytest.fixture(scope="module")
def model_path(model_path):
    assert model_path is not None, "--model-path required"
    return model_path


@pytest.fixture(scope="module")
def det_model(model_path):
    return IndraOCRDetection.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def rec_model(model_path):
    return IndraOCRRecognition.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def group_models(model_path):
    return {
        variant: IndraOCRRecognition.from_pretrained(model_path, variant=variant).eval() for variant in ("cjk", "rtl", "world")
    }


@pytest.fixture(scope="module")
def orient_model(model_path):
    return IndraOCROrientation.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def script_model(model_path):
    return IndraOCRScript.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def order_model(model_path):
    return IndraOCROrder.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def runtime(reference_path):
    if reference_path is None:
        pytest.skip("--reference-path required for the onnx comparison")
    onnxruntime = pytest.importorskip("onnxruntime")
    return {
        name: onnxruntime.InferenceSession(os.path.join(reference_path, name), providers=["CPUExecutionProvider"])
        for name in (
            "det.onnx",
            "rec.onnx",
            "rec_cjk.onnx",
            "rec_rtl.onnx",
            "rec_world.onnx",
            "orient.onnx",
            "script.onnx",
            "order.onnx",
        )
    }


def make_lines(batch, height, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, 1, height, width, generator=generator)
    widths = torch.tensor([width - 28 * i for i in range(batch)], dtype=torch.int64)
    return x, widths


class TestIndraOCRRecognition:
    def test_config_drives_the_architecture(self, rec_model):
        assert rec_model.config.vocab_size == 1698
        assert sum(p.numel() for p in rec_model.parameters()) == 1250310

    def test_group_variants_use_the_same_architecture_without_context(self, group_models):
        assert [model.context is None for model in group_models.values()]
        assert [model.config.vocab_size for model in group_models.values()] == [11750, 1356, 2936]
        assert [sum(p.numel() for p in model.parameters()) for model in group_models.values()] == [
            2535398,
            1194572,
            1398392,
        ]

    @pytest.mark.parametrize("batch,width", [(1, 320), (2, 200), (3, 137)])
    def test_output_shapes(self, rec_model, batch, width):
        x, widths = make_lines(batch, 96, width)
        with torch.no_grad():
            logits = rec_model(x, widths)
            got = rec_model.recognize(x, widths)

        assert logits.shape == (batch, width // 4, 1698)
        assert got["ids"].shape == (batch, width // 4)
        assert got["conf"].shape == (batch, width // 4)
        assert got["topv"].shape == (batch, width // 4, 8)
        assert got["topi"].shape == (batch, width // 4, 8)

    def test_group_output_shapes(self, group_models):
        x, widths = make_lines(2, 32, 200)
        with torch.no_grad():
            for model in group_models.values():
                logits = model(x, widths)
                assert logits.shape == (2, 50, model.config.vocab_size)

    def test_lengths_cover_the_width(self, rec_model):
        widths = torch.tensor([100, 101, 104, 400], dtype=torch.int64)
        assert rec_model.lengths(widths, 26).tolist() == [25, 26, 26, 26]

    def test_decoding_follows_the_logits(self, rec_model):
        x, widths = make_lines(1, 96, 160)
        with torch.no_grad():
            logits = rec_model(x, widths)
            got = rec_model.recognize(x, widths)

        log_probs = logits.log_softmax(-1)
        assert torch.equal(got["ids"], logits.argmax(-1))
        assert torch.equal(got["topi"], log_probs.topk(8, dim=-1).indices)
        assert torch.allclose(got["topv"], log_probs.topk(8, dim=-1).values, atol=1e-5)
        assert torch.allclose(got["conf"], logits.softmax(-1).amax(-1), atol=1e-6)
        assert torch.allclose(got["blank"], log_probs[..., 0], atol=1e-6)

    def test_padded_steps_collapse_to_the_head_bias(self, rec_model):
        x, widths = make_lines(2, 96, 200)
        with torch.no_grad():
            logits = rec_model(x, widths)

        lengths = rec_model.lengths(widths, logits.shape[1])
        assert lengths.tolist() == [50, 43]

        padded = logits[1, lengths[1] :]
        assert torch.allclose(padded, rec_model.head.bias.expand_as(padded), atol=1e-6)

    def test_recognizer_matches_onnx(self, rec_model, runtime):
        x, widths = make_lines(2, 96, 160, seed=1)
        with torch.no_grad():
            got = rec_model.recognize(x, widths)
        expected = runtime["rec.onnx"].run(
            ["ids", "conf", "lengths", "blank", "topv", "topi"], {"x": x.numpy(), "widths": widths.numpy()}
        )
        for name, value in zip(["ids", "conf", "lengths", "blank", "topv", "topi"], expected):
            if name in ("ids", "lengths"):
                np.testing.assert_array_equal(got[name].numpy(), value)
            else:
                np.testing.assert_allclose(got[name].numpy(), value, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("variant", ["cjk", "rtl", "world"])
    def test_group_recognizer_matches_onnx(self, group_models, runtime, variant):
        model = group_models[variant]
        x, widths = make_lines(2, 32, 200, seed=1)
        with torch.no_grad():
            got = model.recognize(x, widths)
        expected = runtime[f"rec_{variant}.onnx"].run(
            ["ids", "conf", "lengths", "blank", "topv", "topi"], {"x": x.numpy(), "widths": widths.numpy()}
        )
        for name, value in zip(["ids", "conf", "lengths", "blank", "topv", "topi"], expected):
            if name in ("ids", "lengths"):
                np.testing.assert_array_equal(got[name].numpy(), value)
            else:
                np.testing.assert_allclose(got[name].numpy(), value, atol=1e-4, rtol=1e-4)


class TestIndraOCRDetection:
    def test_config_drives_the_architecture(self, det_model):
        assert sum(p.numel() for p in det_model.parameters()) == 90568

    @pytest.mark.parametrize("height,width", [(256, 256), (192, 320)])
    def test_output_shapes(self, det_model, height, width):
        generator = torch.Generator().manual_seed(0)
        x = torch.rand(1, 1, height, width, generator=generator)
        with torch.no_grad():
            out = det_model(x)

        assert out.shape == (1, 3, height // 2, width // 2)

    def test_distances_are_positive(self, det_model):
        generator = torch.Generator().manual_seed(0)
        x = torch.rand(1, 1, 64, 96, generator=generator)
        with torch.no_grad():
            out = det_model(x)

        assert out[:, 1:].min() > 0

    def test_detector_matches_onnx(self, det_model, runtime):
        generator = torch.Generator().manual_seed(1)
        x = torch.rand(1, 1, 256, 320, generator=generator)
        with torch.no_grad():
            actual = det_model(x).numpy()
        expected = runtime["det.onnx"].run(["out"], {"x": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


class TestIndraOCROrientation:
    def test_config_drives_the_architecture(self, orient_model):
        assert sum(p.numel() for p in orient_model.parameters()) == 89589

    def test_output_is_a_distribution(self, orient_model):
        generator = torch.Generator().manual_seed(0)
        x = torch.rand(2, 1, 768, 768, generator=generator)
        with torch.no_grad():
            probs = orient_model(x)

        assert probs.shape == (2, 5)
        assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5)

    def test_orientation_matches_onnx(self, orient_model, runtime):
        generator = torch.Generator().manual_seed(1)
        x = torch.rand(2, 1, 768, 768, generator=generator)
        with torch.no_grad():
            actual = orient_model(x).numpy()
        expected = runtime["orient.onnx"].run(["p"], {"x": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


class TestIndraOCRScript:
    def test_config_drives_the_architecture(self, script_model):
        assert sum(p.numel() for p in script_model.parameters()) == 46644

    def test_output_is_a_distribution(self, script_model):
        x, widths = make_lines(3, 32, 300)
        with torch.no_grad():
            probs = script_model(x, widths)

        assert probs.shape == (3, 4)
        assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-5)

    def test_script_matches_onnx(self, script_model, runtime):
        x, widths = make_lines(3, 32, 300, seed=1)
        with torch.no_grad():
            actual = script_model(x, widths).numpy()
        expected = runtime["script.onnx"].run(["p"], {"x": x.numpy(), "widths": widths.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


class TestIndraOCROrder:
    def test_config_drives_the_architecture(self, order_model):
        assert sum(p.numel() for p in order_model.parameters()) == 731658

    def test_output_shapes(self, order_model):
        generator = torch.Generator().manual_seed(0)
        feats = torch.rand(1, 9, 16, generator=generator)
        mask = torch.ones(1, 9, dtype=torch.bool)
        with torch.no_grad():
            a, cls, same, tab = order_model(feats, mask)

        assert a.shape == (1, 9, 9)
        assert cls.shape == (1, 9, 10)
        assert same.shape == (1, 9, 9)
        assert tab.shape == (1, 3, 9, 9)

    def test_pairwise_logits_are_antisymmetric(self, order_model):
        generator = torch.Generator().manual_seed(0)
        feats = torch.rand(1, 7, 16, generator=generator)
        mask = torch.ones(1, 7, dtype=torch.bool)
        with torch.no_grad():
            a, _, same, _ = order_model(feats, mask)

        assert torch.allclose(a + a.transpose(-1, -2), torch.zeros_like(a), atol=1e-5)
        assert torch.allclose(same, same.transpose(-1, -2), atol=1e-6)

    def test_order_matches_onnx(self, order_model, runtime):
        generator = torch.Generator().manual_seed(1)
        feats = torch.rand(1, 9, 16, generator=generator)
        mask = torch.ones(1, 9, dtype=torch.bool)
        with torch.no_grad():
            actual = [value.numpy() for value in order_model(feats, mask)]
        expected = runtime["order.onnx"].run(["a", "cls", "same", "tab"], {"feats": feats.numpy(), "mask": mask.numpy()})
        for got, value in zip(actual, expected):
            np.testing.assert_allclose(got, value, atol=1e-4, rtol=1e-4)


class TestIndraOCRLanguageModel:
    @pytest.fixture(scope="class")
    def language_model(self, model_path):
        return NgramLM(os.path.join(model_path, "lm.npz"))

    def test_table_is_read_from_the_npz(self, language_model):
        assert language_model.order == 5
        assert language_model.vocab_size == 1678
        assert language_model.bits == 11

    def test_batched_lookup_matches_single_lookup(self, language_model):
        history = (12, 34, 56, 78)
        token = 90
        key = (_pack(history, language_model.bits) << language_model.bits) | (token + 1)
        assert language_model.logp(history, token) == language_model.logp_keys([key])[0]

    def test_beam_search_returns_a_reading_per_step(self, language_model):
        rng = np.random.default_rng(0)
        blank = np.linspace(-0.1, -6.0, 16)
        topi = rng.integers(0, language_model.vocab_size + 1, size=(16, 8)).astype(np.int64)
        topv = np.log(rng.uniform(0.001, 1.0, size=(16, 8)))
        ids, steps = beam_search(blank, topv, topi, language_model, cand_floor=3e-3)

        assert len(ids) == len(steps)
        assert all(0 <= step < len(blank) for step in steps)
        assert all(0 < identifier <= language_model.vocab_size for identifier in ids)
