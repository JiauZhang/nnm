import torch

from tests.models.smoke_registry import SmokeCase, from_config, shape
from tests.models.test_minicpm_ocr import tiny_config

CASES = [
    SmokeCase(
        "minicpm-ocr",
        "text",
        from_config("nnm.models.minicpm_ocr.model", "MiniCPMOcrModel", tiny_config),
        lambda: (torch.randint(1, 90, (2, 9)),),
        shape(2, 9, 100),
    ),
]
