import pytest

from tests.models.smoke_audio import CASES as AUDIO
from tests.models.smoke_diffusion import CASES as DIFFUSION
from tests.models.smoke_llm import CASES as LLM
from tests.models.smoke_ocr import CASES as OCR
from tests.models.smoke_registry import run_case
from tests.models.smoke_vision import CASES as VISION

CASES = DIFFUSION + VISION + AUDIO + LLM + OCR


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_model_smoke(case):
    run_case(case)
