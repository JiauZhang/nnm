import os

import pytest
import torch

from nnm.models.indra_ocr.det import IndraOCRDetection
from nnm.models.indra_ocr.rec import IndraOCRRecognition


@pytest.fixture(scope="module")
def model_path(model_path):
    assert model_path is not None, "--model-path required"
    return model_path


@pytest.fixture(scope="module")
def rec_model(model_path):
    return IndraOCRRecognition.from_pretrained(os.path.join(model_path, "rec")).eval()


@pytest.fixture(scope="module")
def det_model(model_path):
    return IndraOCRDetection.from_pretrained(os.path.join(model_path, "det")).eval()


def make_lines(batch, width):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(batch, 1, 96, width, generator=generator)
    widths = torch.tensor([width - 28 * i for i in range(batch)], dtype=torch.int64)
    return x, widths


class TestIndraOCRRecognition:
    def test_config_drives_the_architecture(self, rec_model):
        assert rec_model.config.vocab_size == 1698
        assert sum(p.numel() for p in rec_model.parameters()) == 1250310

    @pytest.mark.parametrize("batch,width", [(1, 320), (2, 200), (3, 137)])
    def test_output_shapes(self, rec_model, batch, width):
        x, widths = make_lines(batch, width)
        with torch.no_grad():
            logits = rec_model(x, widths)
            got = rec_model.recognize(x, widths)

        assert logits.shape == (batch, width // 4, 1698)
        assert got["ids"].shape == (batch, width // 4)
        assert got["conf"].shape == (batch, width // 4)
        assert got["topv"].shape == (batch, width // 4, 8)
        assert got["topi"].shape == (batch, width // 4, 8)

    def test_lengths_cover_the_width(self, rec_model):
        widths = torch.tensor([100, 101, 104, 400], dtype=torch.int64)
        assert rec_model.lengths(widths, 26).tolist() == [25, 26, 26, 26]

    def test_decoding_follows_the_logits(self, rec_model):
        x, widths = make_lines(1, 160)
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
        x, widths = make_lines(2, 200)
        with torch.no_grad():
            logits = rec_model(x, widths)

        lengths = rec_model.lengths(widths, logits.shape[1])
        assert lengths.tolist() == [50, 43]

        padded = logits[1, lengths[1] :]
        assert torch.allclose(padded, rec_model.head.bias.expand_as(padded), atol=1e-6)


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
