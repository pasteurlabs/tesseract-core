import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options for pytest."""
    parser.addoption(
        "--dir", action="store", default=".", help="Directory of your tesseract api"
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Get/set command line arguments."""
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.name
    if "dir" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("tesseract_dir", [option_value])


@pytest.fixture
def bootcamp_image_name() -> str:
    """Expected image name for bootcamp."""
    return "bootcamp"
