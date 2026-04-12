import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default=None,
        help="Path to the pretrained model directory",
    )


@pytest.fixture
def model_path(request):
    return request.config.getoption("--model-path")
