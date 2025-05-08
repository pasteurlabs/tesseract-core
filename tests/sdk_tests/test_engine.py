# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import time
from pathlib import Path

import pytest
import yaml
from jinja2.exceptions import TemplateNotFound
from typeguard import suppress_type_checks

from tesseract_core.sdk import engine
from tesseract_core.sdk.api_parse import (
    TesseractConfig,
    validate_tesseract_api,
)
from tesseract_core.sdk.cli import AVAILABLE_RECIPES
from tesseract_core.sdk.docker_client import Image
from tesseract_core.sdk.exceptions import UserError


def test_prepare_build_context(tmp_path_factory):
    """Test we can create a dockerfile."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "foo").touch()
    build_dir = tmp_path_factory.mktemp("build")
    default_config = TesseractConfig(name="foobar")
    engine.prepare_build_context(src_dir, build_dir, default_config)
    assert (build_dir / "__tesseract_source__" / "foo").exists()
    assert (build_dir / "__tesseract_runtime__").exists()
    assert (build_dir / "Dockerfile").exists()


@pytest.mark.parametrize("generate_only", [True, False])
def test_build_tesseract(dummy_tesseract_package, mocked_docker, generate_only, caplog):
    """Test we can build an image for a package and keep build directory."""
    src_dir = dummy_tesseract_package
    image_name = "unit_vectoradd"
    image_tag = "42"

    with caplog.at_level(logging.INFO):
        out = engine.build_tesseract(
            src_dir,
            image_tag,
            generate_only=generate_only,
        )

    if generate_only:
        assert isinstance(out, Path)

        # Check if Dockerfile is there
        dockerfile_path = out / "Dockerfile"
        assert dockerfile_path.exists()

        # Check stdout if it contains the correct docker build command
        assert "docker buildx build" in caplog.text
        assert str(out) in caplog.text
    else:
        assert isinstance(out, Image)
        assert out.attrs == mocked_docker.images.get(image_name).attrs


@pytest.mark.parametrize("recipe", [None, *AVAILABLE_RECIPES])
def test_init(tmpdir, recipe):
    """Test the initialization of a tesseract from the template."""
    if recipe:
        api_path = engine.init_api(
            target_dir=Path(tmpdir) / "test_dir", tesseract_name="foo", recipe=recipe
        )
    else:
        api_path = engine.init_api(
            target_dir=Path(tmpdir) / "test_dir", tesseract_name="foo"
        )

    # Make sure that all tesseract related files are created
    assert api_path.exists()
    assert (tmpdir / "test_dir/tesseract_requirements.txt").exists()
    assert (tmpdir / "test_dir/tesseract_config.yaml").exists()

    # Ensure the name in the config is correct
    with open(tmpdir / "test_dir/tesseract_config.yaml") as config_yaml:
        test = yaml.safe_load(config_yaml)
        assert test["name"] == "foo"

    # Ensure template passes validation
    validate_tesseract_api(api_path.parent)

    if not recipe:
        # Ensure it still passes when commenting in optional endpoints
        with open(api_path) as f:
            api_code = f.read()

        start_idx = api_code.find("# Optional endpoints")
        assert start_idx != -1

        api_code = api_code[: start_idx + 1] + api_code[start_idx + 1 :].replace(
            "# ", ""
        )

        with open(api_path, "w") as f:
            f.write(api_code)

    validate_tesseract_api(api_path.parent)


def test_init_bad_recipe(tmpdir):
    """Test the initialization of a tesseract with a bad recipe.

    This does not check for pretty terminal output or typer validation.
    But ensures that an error is raised if there are missing template files.
    """
    with pytest.raises(TemplateNotFound):
        engine.init_api(
            target_dir=Path(tmpdir) / "test_dir",
            tesseract_name="foo",
            recipe="recipewillneverexist",
        )


def test_run_tesseract(mocked_docker):
    """Test running a tesseract."""
    res_out, res_err = engine.run_tesseract(
        "foobar", "apply", ['{"inputs": {"a": [1, 2, 3], "b": [4, 5, 6]}}']
    )

    # Mocked docker just returns the kwargs to `docker run` as json
    res = json.loads(res_out)
    assert res["command"] == ["apply", '{"inputs": {"a": [1, 2, 3], "b": [4, 5, 6]}}']
    assert res["image"] == "foobar"

    # Also check that stderr is captured
    assert res_err == "hello tesseract"

    # Check that we did not request GPUs by accident
    assert res["device_requests"] is None


def test_run_gpu(mocked_docker):
    """Test running a tesseract with all available GPUs."""
    res_out, _ = engine.run_tesseract(
        "foobar",
        "apply",
        ['{"inputs": {"a": [1, 2, 3], "b": [4, 5, 6]}}'],
        gpus=["all"],
    )

    res = json.loads(res_out)
    assert res["device_requests"] == ["all"]


def test_run_tesseract_file_input(mocked_docker, tmpdir):
    """Test running a tesseract with file input / output."""
    outdir = Path(tmpdir) / "output"
    outdir.mkdir()

    infile = Path(tmpdir) / "input.json"
    infile.touch()

    res, _ = engine.run_tesseract(
        "foobar",
        "apply",
        [f"@{infile}", "--output-path", str(outdir)],
    )

    # Mocked docker just returns the kwargs to `docker run` as json
    res = json.loads(res)
    assert res["command"] == [
        "apply",
        "@/mnt/payload.json",
        "--output-path",
        "/mnt/output",
    ]
    assert res["image"] == "foobar"
    assert res["volumes"].keys() == {str(infile), str(outdir)}

    # Test the same with a folder mount
    res, _ = engine.run_tesseract(
        "foobar",
        "apply",
        [f"@{infile}", "--output-path", str(outdir)],
        volumes=["/path/on/host:/path/in/container:ro"],
    )
    res = json.loads(res)
    assert res["volumes"].keys() == {str(infile), str(outdir), "/path/on/host"}
    assert res["volumes"]["/path/on/host"] == {
        "mode": "ro",
        "bind": "/path/in/container",
    }


def test_serve_tesseracts_invalid_input_args():
    """Test input validation logic for multi-tesseract serve."""
    with suppress_type_checks():
        with pytest.raises(ValueError):
            engine.serve([None])

        with pytest.raises(ValueError):
            engine.serve([])

        with pytest.raises(ValueError):
            engine.serve([None, "vectoradd"])

        with pytest.raises(ValueError):
            engine.teardown(None)


def test_get_tesseract_images(mocked_docker):
    tesseract_images = engine.get_tesseract_images()
    assert len(tesseract_images) == 2


def test_get_tesseract_containers(mocked_docker):
    tesseract_containers = engine.get_tesseract_containers()
    assert len(tesseract_containers) == 1


def test_serve_tesseracts(mocked_docker):
    """Test multi-tesseract serve."""
    # Serve valid
    project_name_single_tesseract = engine.serve(["vectoradd"])
    assert project_name_single_tesseract

    # Teardown valid
    engine.teardown(project_name_single_tesseract)

    # Multi-serve valid
    project_name_multi_tesseract = engine.serve(["vectoradd", "vectoradd"])
    assert project_name_multi_tesseract

    # Multi-teardown valid
    engine.teardown(project_name_multi_tesseract)

    # Tear down invalid
    with pytest.raises(ValueError):
        engine.teardown("invalid_project_name")

    # Serve with gpus
    project_name_multi_tesseract = engine.serve(["vectoradd"], gpus=["1", "3"])
    assert project_name_multi_tesseract


def test_needs_docker(mocked_docker, monkeypatch):
    @engine.needs_docker
    def run_something_with_docker():
        pass

    # Happy case
    run_something_with_docker()

    # Sad case
    def raise_docker_error(*args, **kwargs):
        raise RuntimeError("No Docker")

    monkeypatch.setattr(mocked_docker, "info", raise_docker_error)

    with pytest.raises(UserError):
        run_something_with_docker()


def test_logpipe(caplog):
    # Verify that logging in a separate thread works as intended
    from tesseract_core.sdk.engine import LogPipe

    caplog.set_level(logging.INFO, logger="tesseract")

    logged_lines = []
    for _ in range(100):
        msg_length = 2 ** random.randint(1, 12)
        msg = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=msg_length))
        logged_lines.append(msg)

    logpipe = LogPipe(logging.INFO)
    with logpipe:
        fd = os.fdopen(logpipe.fileno(), "w", closefd=False)
        for line in logged_lines:
            print(line, file=fd)
            time.sleep(random.random() / 100)
        fd.flush()

    assert logpipe.captured_lines == logged_lines
    assert caplog.record_tuples == [
        ("tesseract", logging.INFO, line) for line in logged_lines
    ]


def test_parse_requirements(tmpdir):
    reqs = """
    --extra-index-url https://download.pytorch.org/whl/cpu
    torch==2.5.1

    --find-links https://data.pyg.org/whl/torch-2.5.1+cpu.html
    torch_scatter==2.1.2+pt25cpu

    ./internal_packages/foobar
    """
    reqs_file = Path(tmpdir) / "requirements.txt"
    with open(reqs_file, "w") as fi:
        fi.write(reqs)
    locals, remotes = engine.parse_requirements(reqs_file)

    assert locals == [
        "./internal_packages/foobar",
    ]
    assert remotes == [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "torch==2.5.1",
        "--find-links https://data.pyg.org/whl/torch-2.5.1+cpu.html",
        "torch_scatter==2.1.2+pt25cpu",
    ]
