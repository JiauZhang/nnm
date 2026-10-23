import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to the pretrained model directory",
    )
    parser.addoption(
        "--reference-path",
        action="store",
        default=None,
        help="Path to the upstream checkpoint a converted nnm model was built from",
    )


@pytest.fixture(scope="module")
def model_path(request):
    return request.config.getoption("--model-path")


@pytest.fixture(scope="module")
def reference_path(request):
    return request.config.getoption("--reference-path")
