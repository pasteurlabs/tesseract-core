"""Tests for the Tesseract Bootcamp Tutorial."""

import yaml
from typer.testing import CliRunner

from tesseract_core.sdk.cli import app

cli_runner = CliRunner(mix_stderr=False)


def test_00_tesseract_init(tesseract_dir: str, bootcamp_image_name: str) -> None:
    """Test for step 0 of the bootcamp tutorial."""
    # Check that the Tesseract named "bootcamp" exists
    # and can be run

    assert (tesseract_dir / "tesseract_api.py").exists()
    with open(tesseract_dir / "tesseract_config.yaml") as config_yaml:
        assert yaml.safe_load(config_yaml)["name"] == "bootcamp"

    test_commands = ("input-schema", "output-schema", "openapi-schema", "health")
    for command in test_commands:
        run_res = cli_runner.invoke(
            app,
            [
                "run",
                bootcamp_image_name,
                command,
            ],
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        assert run_res.stdout


def test_02_tesseract_apply(bootcamp_image_name: str) -> None:
    """Test for step 2 of the bootcamp tutorial."""
    # Check that the apply function is implemented
    # Convert numbers to words
    run_res = cli_runner.invoke(
        app,
        ["run", bootcamp_image_name, "apply", "@./example_data/test_apply.json"],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    # Check stdout contains the expected output
    # "Hello Tesseract"
    assert "Hello Tesseract" in run_res.stdout


def test_03_tesseract_localpackage(bootcamp_image_name: str) -> None:
    """Test for step 3 of the bootcamp tutorial."""
    assert True
