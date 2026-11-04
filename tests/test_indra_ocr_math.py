import os

import numpy as np
import pytest
import torch

from nnm.models.indra_ocr.math import IndraOCRMath


@pytest.fixture(scope="module")
def model(model_path):
    assert model_path is not None, "--model-path required"
    return IndraOCRMath.from_pretrained(model_path).eval()


@pytest.fixture(scope="module")
def runtime(reference_path):
    if reference_path is None:
        pytest.skip("--reference-path required for the onnx comparison")
    onnxruntime = pytest.importorskip("onnxruntime")
    return {
        "classifier": onnxruntime.InferenceSession(
            os.path.join(reference_path, "mathcls.onnx"), providers=["CPUExecutionProvider"]
        ),
        "encoder": onnxruntime.InferenceSession(
            os.path.join(reference_path, "math_enc.onnx"), providers=["CPUExecutionProvider"]
        ),
        "init": onnxruntime.InferenceSession(
            os.path.join(reference_path, "math_dec_init.onnx"), providers=["CPUExecutionProvider"]
        ),
        "step": onnxruntime.InferenceSession(
            os.path.join(reference_path, "math_dec_step.onnx"), providers=["CPUExecutionProvider"]
        ),
    }


def make_line(batch, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, 64, width, generator=generator) * 0.1


class TestIndraOCRMath:
    def test_config_drives_the_architecture(self, model):
        assert model.config.encoder.channels == 192
        assert model.config.encoder.heads == 8
        assert model.config.decoder.vocab_size == 652
        assert sum(parameter.numel() for parameter in model.parameters()) == 4596829

    def test_classifier_scores_lines(self, model):
        x = make_line(2, 128)
        widths = torch.tensor([128, 64])
        with torch.no_grad():
            scores = model.classifier(x[:, :, :32], widths)

        assert scores.shape == (2,)
        assert torch.all((scores > 0) & (scores < 1))

    def test_flags_threshold_the_classifier(self, model):
        x = make_line(2, 128)
        widths = torch.tensor([128, 64])
        with torch.no_grad():
            flags = model.flags(x[:, :, :32], widths)

        assert flags.shape == (2,)
        assert flags.dtype == torch.bool

    def test_detokenize_joins_math_tokens_with_spaces(self, model):
        assert model.detokenize(["a", "$", "x", "+", "1", "$", "b"]) == "a$x + 1$b"
        assert model.detokenize(["$", "{", "x", "}"]) == "${ x }$"

    def test_decode_reads_each_line(self, model):
        x = make_line(1, 160)
        widths = torch.tensor([160])
        with torch.no_grad():
            readings = model.decode(x, widths)

        assert len(readings) == 1
        text, confidence = readings[0]
        assert isinstance(text, str)
        assert 0.0 <= confidence <= 1.0

    def test_encoder_masks_the_padded_steps(self, model):
        x = make_line(2, 160)
        widths = torch.tensor([160, 120])
        with torch.no_grad():
            memory, mask = model.encode(x, widths)

        assert memory.shape == (2, 20, 192)
        assert mask.shape == (2, 20)
        assert mask[0].sum() == 20
        assert mask[1].sum() == 15

    def test_decoder_runs_incrementally(self, model):
        memory = torch.randn(2, 20, 192)
        mask = torch.ones(2, 20)
        mask[1, 15:] = 0
        with torch.no_grad():
            cross_key, cross_value = model.decoder.encode_memory(memory)
            self_key, self_value = model.decoder.empty_cache(2, memory.device, memory.dtype)
            logits, key, value = model.decoder(
                torch.tensor([5, 7]), torch.tensor(0), self_key, self_value, cross_key, cross_value, mask
            )
            next_logits, next_key, _ = model.decoder(
                torch.tensor([9, 3]), torch.tensor(1), key, value, cross_key, cross_value, mask
            )

        assert logits.shape == (2, 652)
        assert next_logits.shape == (2, 652)
        assert key.shape == (3, 2, 8, 1, 24)
        assert next_key.shape == (3, 2, 8, 2, 24)

    def test_classifier_matches_onnx(self, model, runtime):
        x = make_line(2, 160)
        widths = torch.tensor([160, 90])
        with torch.no_grad():
            actual = model.classifier(x[:, :, :32], widths).numpy()
        expected = runtime["classifier"].run(["p"], {"x": x[:, :, :32].numpy(), "widths": widths.numpy()})[0]
        np.testing.assert_allclose(actual, expected.reshape(-1), atol=1e-5, rtol=1e-5)

    def test_encoder_matches_onnx(self, model, runtime):
        x = make_line(2, 160)
        widths = torch.tensor([160, 120])
        with torch.no_grad():
            memory, mask = model.encode(x, widths)
        expected_memory, expected_mask = runtime["encoder"].run(["memory", "mask"], {"x": x.numpy(), "widths": widths.numpy()})
        np.testing.assert_allclose(memory.numpy(), expected_memory, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(mask.numpy(), expected_mask)

    def test_decoder_matches_onnx(self, model, runtime):
        memory = runtime["encoder"].run(["memory"], {"x": make_line(2, 160).numpy(), "widths": np.array([160, 120])})[0]
        mask = runtime["encoder"].run(["mask"], {"x": make_line(2, 160).numpy(), "widths": np.array([160, 120])})[0]
        expected_key, expected_value = runtime["init"].run(["cross_k", "cross_v"], {"memory": memory})
        with torch.no_grad():
            cross_key, cross_value = model.decoder.encode_memory(torch.tensor(memory))
            self_key, self_value = model.decoder.empty_cache(2, torch.device("cpu"), torch.float32)
            logits, key, value = model.decoder(
                torch.tensor([5, 7]), torch.tensor(0), self_key, self_value, cross_key, cross_value, torch.tensor(mask)
            )

        np.testing.assert_allclose(cross_key.numpy(), expected_key, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cross_value.numpy(), expected_value, atol=1e-5, rtol=1e-5)
        expected_logits, expected_key_step, expected_value_step = runtime["step"].run(
            ["logits", "new_k", "new_v"],
            {
                "token": np.array([5, 7], dtype=np.int64),
                "pos": np.array(0, dtype=np.int64),
                "self_k": self_key.numpy(),
                "self_v": self_value.numpy(),
                "cross_k": expected_key,
                "cross_v": expected_value,
                "mask": mask,
            },
        )
        np.testing.assert_allclose(logits.numpy(), expected_logits, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(key.numpy(), expected_key_step, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(value.numpy(), expected_value_step, atol=1e-5, rtol=1e-5)
