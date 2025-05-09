"""Tests for the Tesseract Bootcamp Tutorial."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from tesseract_core.sdk.api_parse import (
    validate_tesseract_api,
)
from tesseract_core.sdk.cli import app

cli_runner = CliRunner(mix_stderr=False)
BOOTCAMP_IMAGE_NAME = "bootcamp"


def test_00_tesseract_init(tesseract_dir: Path) -> None:
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
                BOOTCAMP_IMAGE_NAME,
                command,
            ],
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        assert run_res.stdout


def test_01a_tesseract_schema(tesseract_dir: Path) -> None:
    """Test for step 1 of the bootcamp tutorial.

    Validate the input and output schema of the Tesseract.
    """
    validate_tesseract_api(tesseract_dir)

    run_res = cli_runner.invoke(
        app,
        [
            "build",
            tesseract_dir,
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    test_commands = ("input-schema", "output-schema")
    for command in test_commands:
        run_res = cli_runner.invoke(
            app,
            [
                "run",
                BOOTCAMP_IMAGE_NAME,
                command,
            ],
            catch_exceptions=False,
        )
        assert run_res.exit_code == 0, run_res.stderr
        assert run_res.stdout
        # Validate that the input schema is array and output is string
        if command == "input-schema":
            assert "EncodedArrayModel" in run_res.stdout
        elif command == "output-schema":
            assert '"type":"string"' in run_res.stdout


def test_01b_tesseract_apply(tesseract_dir: Path) -> None:
    """Test for step 2 of the bootcamp tutorial.

    Validate that the apply function is correctly implemented.
    """
    run_res = cli_runner.invoke(
        app,
        [
            "build",
            tesseract_dir,
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    # Check that the apply function is implemented
    # Convert numbers to words
    data_dir = Path(__file__).parent / "example_data"
    run_res = cli_runner.invoke(
        app,
        ["run", BOOTCAMP_IMAGE_NAME, "apply", f"@{data_dir}/test_apply.json"],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr
    # Check stdout contains the expected output
    # "Hello Tesseract"
    assert "hello tesseract" in run_res.stdout


def test_02_tesseract_packagedata() -> None:
    """Test for step 3 of the bootcamp tutorial.

    Call the apply function on secret_message.json and check that
    the output is correct.
    """
    data_dir = Path(__file__).parent / "example_data"
    run_res = cli_runner.invoke(
        app,
        ["run", BOOTCAMP_IMAGE_NAME, "apply", f"@{data_dir}/secret_message.json"],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0, run_res.stderr

    assert (
        "and remember dana there is such a thing in the real world as a tesseract and it works"
        in run_res.stdout
    )
