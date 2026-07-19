# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import socket
import time
from contextlib import closing
from pathlib import Path

import pytest
import yaml
from jinja2.exceptions import TemplateNotFound

from tesseract_core.sdk import engine
from tesseract_core.sdk.api_parse import (
    TesseractBuildConfig,
    TesseractConfig,
    validate_tesseract_api,
)
from tesseract_core.sdk.cli import AVAILABLE_RECIPES
from tesseract_core.sdk.docker_client import Image, NotFound
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


def test_prepare_build_context_python_version(tmp_path_factory):
    """Test that python_version is rendered as ENV in the Dockerfile."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "foo").touch()
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(
        name="foobar",
        build_config=TesseractBuildConfig(python_version="3.12"),
    )
    engine.prepare_build_context(src_dir, build_dir, config)

    dockerfile = (build_dir / "Dockerfile").read_text()
    assert 'TESSERACT_PYTHON_VERSION="3.12"' in dockerfile

    # Without python_version, the env var should not appear
    build_dir2 = tmp_path_factory.mktemp("build2")
    config_default = TesseractConfig(name="foobar")
    engine.prepare_build_context(src_dir, build_dir2, config_default)

    dockerfile_default = (build_dir2 / "Dockerfile").read_text()
    assert "TESSERACT_PYTHON_VERSION" not in dockerfile_default


def test_prepare_build_context_env(tmp_path_factory):
    """Test that env variables are rendered as ENV lines in the Dockerfile."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "tesseract_api.py").touch()
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(
        name="foobar",
        env={
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "MY_VAR": "hello world",
        },
    )

    engine.prepare_build_context(src_dir, build_dir, config)
    dockerfile = (build_dir / "Dockerfile").read_text()
    assert 'ENV XLA_PYTHON_CLIENT_PREALLOCATE="false"' in dockerfile
    assert 'ENV MY_VAR="hello world"' in dockerfile


def test_prepare_build_context_external_package_data(tmp_path_factory):
    """Test package_data from outside the Tesseract directory is copied correctly."""
    # Create a parent directory with src and external subdirectories
    parent_dir = tmp_path_factory.mktemp("parent")
    src_dir = parent_dir / "tesseract"
    src_dir.mkdir()
    (src_dir / "tesseract_api.py").touch()

    # Create external files (sibling to src_dir)
    external_dir = parent_dir / "external"
    external_dir.mkdir()
    external_file = external_dir / "shared_code.py"
    external_file.write_text("# shared code")

    build_dir = tmp_path_factory.mktemp("build")

    # Configure package_data with relative external path
    config = TesseractConfig(
        name="foobar",
        build_config=TesseractBuildConfig(
            package_data=[("../external/shared_code.py", "shared_code.py")],
        ),
    )

    engine.prepare_build_context(src_dir, build_dir, config)

    # Verify external file was copied to __package_data__
    assert (build_dir / "__package_data__" / "shared_code.py").exists()
    assert (
        build_dir / "__package_data__" / "shared_code.py"
    ).read_text() == "# shared code"


def test_prepare_build_context_package_data_same_basename(tmp_path_factory):
    """Test that package_data with same source filenames but different targets works."""
    # Create a parent directory with src and two external subdirectories
    parent_dir = tmp_path_factory.mktemp("parent")
    src_dir = parent_dir / "tesseract"
    src_dir.mkdir()
    (src_dir / "tesseract_api.py").touch()

    # Create two external directories with files of the same name
    external_dir1 = parent_dir / "external1"
    external_dir2 = parent_dir / "external2"
    external_dir1.mkdir()
    external_dir2.mkdir()
    (external_dir1 / "config.yaml").write_text("# config 1")
    (external_dir2 / "config.yaml").write_text("# config 2")

    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(
        name="foobar",
        build_config=TesseractBuildConfig(
            package_data=[
                ("../external1/config.yaml", "config1.yaml"),
                ("../external2/config.yaml", "config2.yaml"),
            ],
        ),
    )

    engine.prepare_build_context(src_dir, build_dir, config)

    # Both files should be copied into the build context
    package_data_dir = build_dir / "__package_data__"
    assert (package_data_dir / "config.yaml").exists()
    assert (package_data_dir / "config_1.yaml").exists()
    assert (package_data_dir / "config.yaml").read_text() == "# config 1"
    assert (package_data_dir / "config_1.yaml").read_text() == "# config 2"

    # Duplicate target paths should raise an error
    config_dup = TesseractConfig(
        name="foobar",
        build_config=TesseractBuildConfig(
            package_data=[
                ("../external1/config.yaml", "same_target.yaml"),
                ("../external2/config.yaml", "same_target.yaml"),
            ],
        ),
    )

    build_dir2 = tmp_path_factory.mktemp("build2")
    with pytest.raises(RuntimeError, match="duplicate target path"):
        engine.prepare_build_context(src_dir, build_dir2, config_dup)


def test_prepare_build_context_package_data_not_found(tmp_path_factory):
    """Test that package_data with non-existent file raises an error."""
    parent_dir = tmp_path_factory.mktemp("parent")
    src_dir = parent_dir / "tesseract"
    src_dir.mkdir()
    (src_dir / "tesseract_api.py").touch()
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(
        name="foobar",
        build_config=TesseractBuildConfig(
            package_data=[("../nonexistent/file.py", "file.py")],
        ),
    )

    with pytest.raises(RuntimeError, match="package_data source file not found"):
        engine.prepare_build_context(src_dir, build_dir, config)


def test_prepare_build_context_local_dependency_with_extras(tmp_path_factory):
    """Local pip dependency with an extras specifier is staged (issue #643)."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "tesseract_api.py").touch()
    (src_dir / "mylocaldep").mkdir()
    (src_dir / "tesseract_requirements.txt").write_text("numpy\n./mylocaldep[extra]\n")
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(name="foobar")
    engine.prepare_build_context(src_dir, build_dir, config)

    # The directory (without the extras suffix) is staged.
    assert (build_dir / "local_requirements" / "mylocaldep").is_dir()
    assert not (build_dir / "local_requirements" / "mylocaldep[extra]").exists()

    # The rewritten requirements file installs it from the staged copy, keeping
    # the extra so pip installs it.
    reqs = (
        (build_dir / "__tesseract_source__" / "tesseract_requirements.txt")
        .read_text()
        .splitlines()
    )
    assert "numpy" in reqs
    assert "./local_requirements/mylocaldep[extra]" in reqs
    # The original local line is rewritten, not carried over verbatim.
    assert "./mylocaldep[extra]" not in reqs


def test_prepare_build_context_local_dependency_file_url(tmp_path_factory):
    """Local pip dependency given as a file:// URL is staged (issue #643)."""
    dep_dir = tmp_path_factory.mktemp("dep") / "mylocaldep"
    dep_dir.mkdir()
    (dep_dir / "setup.py").touch()
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "tesseract_api.py").touch()
    (src_dir / "tesseract_requirements.txt").write_text(f"numpy\nfile://{dep_dir}\n")
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(name="foobar")
    engine.prepare_build_context(src_dir, build_dir, config)

    # The file:// URL is resolved to a native path and staged by its name.
    assert (build_dir / "local_requirements" / "mylocaldep").is_dir()

    reqs = (
        (build_dir / "__tesseract_source__" / "tesseract_requirements.txt")
        .read_text()
        .splitlines()
    )
    assert "numpy" in reqs
    assert "./local_requirements/mylocaldep" in reqs


def test_prepare_build_context_local_dependency_not_found(tmp_path_factory):
    """A local dependency that does not exist raises a clear error."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "tesseract_api.py").touch()
    (src_dir / "tesseract_requirements.txt").write_text("./does_not_exist\n")
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(name="foobar")
    with pytest.raises(RuntimeError, match="local dependency not found"):
        engine.prepare_build_context(src_dir, build_dir, config)


def test_prepare_build_context_local_dependency_parent_path(tmp_path_factory):
    """Local pip dependency reached via ../.. is staged (issue #630)."""
    parent_dir = tmp_path_factory.mktemp("parent")
    # A package that lives two levels above the Tesseract source directory.
    (parent_dir / "mypkg").mkdir()
    (parent_dir / "mypkg" / "setup.py").touch()
    src_dir = parent_dir / "sub" / "tesseract"
    src_dir.mkdir(parents=True)
    (src_dir / "tesseract_api.py").touch()
    (src_dir / "tesseract_requirements.txt").write_text("../../mypkg\n")
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(name="foobar")
    engine.prepare_build_context(src_dir, build_dir, config)

    # The staged name comes from the resolved path, not from `../..`.
    assert (build_dir / "local_requirements" / "mypkg").is_dir()
    assert [p.name for p in (build_dir / "local_requirements").iterdir()] == ["mypkg"]
    reqs = (
        (build_dir / "__tesseract_source__" / "tesseract_requirements.txt")
        .read_text()
        .splitlines()
    )
    assert "./local_requirements/mypkg" in reqs
    assert "../../mypkg" not in reqs


def test_prepare_build_context_conda_local_dependency(tmp_path_factory):
    """Conda provider stages local-path pip dependencies (issue #641)."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "tesseract_api.py").touch()
    (src_dir / "mypkg_src").mkdir()
    (src_dir / "mypkg_src" / "setup.py").touch()
    env_spec = {
        "name": "tesseract",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.12",
            "pip",
            {"pip": ["requests", "./mypkg_src"]},
        ],
    }
    with (src_dir / "tesseract_environment.yaml").open("w") as f:
        yaml.safe_dump(env_spec, f)
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(
        name="foobar",
        build_config=TesseractBuildConfig(
            requirements={"provider": "conda"},
        ),
    )
    engine.prepare_build_context(src_dir, build_dir, config)

    # The local package is staged into the build context.
    assert (build_dir / "local_requirements" / "mypkg_src").is_dir()

    # The rewritten environment file points pip at the staged copy while leaving
    # remote dependencies untouched.
    rewritten = yaml.safe_load(
        (build_dir / "__tesseract_source__" / "tesseract_environment.yaml").read_text()
    )
    pip_entry = next(e["pip"] for e in rewritten["dependencies"] if isinstance(e, dict))
    assert "requests" in pip_entry
    assert "./local_requirements/mypkg_src" in pip_entry
    # The original local path is rewritten, not carried over verbatim.
    assert "./mypkg_src" not in pip_entry


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


def test_run_memory(mocked_docker):
    """Test running a tesseract with memory limit."""
    res_out, _ = engine.run_tesseract(
        "foobar",
        "apply",
        ['{"inputs": {"a": [1, 2, 3], "b": [4, 5, 6]}}'],
        memory="512m",
    )

    res = json.loads(res_out)
    assert res["memory"] == "512m"


def test_run_tesseract_file_input(mocked_docker, tmpdir):
    """Test running a tesseract with file input / output."""
    outdir = Path(tmpdir) / "output"
    outdir.mkdir()

    infile = Path(tmpdir) / "input.json"
    infile.touch()

    res, _ = engine.run_tesseract(
        "foobar",
        "apply",
        [f"@{infile}"],
        output_path=str(outdir),
    )

    # Mocked docker just returns the kwargs to `docker run` as json
    res = json.loads(res)
    assert res["command"] == [
        "apply",
        "@/tesseract/payload.json",
    ]
    assert res["environment"]["TESSERACT_OUTPUT_PATH"] == "/tesseract/output_data"
    assert res["image"] == "foobar"
    assert res["volumes"].keys() == {str(infile), str(outdir)}

    # Test the same with a folder mount
    res, _ = engine.run_tesseract(
        "foobar",
        "apply",
        [f"@{infile}"],
        volumes=[f"{tmpdir}:/path/in/container:ro"],
        output_path=str(outdir),
    )
    res = json.loads(res)
    assert res["volumes"].keys() == {str(infile), str(outdir), f"{tmpdir}"}
    assert res["volumes"][f"{tmpdir}"] == {
        "mode": "ro",
        "bind": "/path/in/container",
    }

    # test that identical source folders raise an error
    with pytest.raises(ValueError):
        res, _ = engine.run_tesseract(
            "foobar",
            "apply",
            [f"@{infile}"],
            volumes=[
                f"{tmpdir}:/path/in/container:ro",
                f"{tmpdir}:/path/in/container2:ro",
            ],
            output_path=str(outdir),
        )

    # Test the same but with --input-path
    indir = tmpdir / "input_path"
    indir.mkdir()
    res, _ = engine.run_tesseract(
        "foobar",
        "apply",
        [f"@{infile}"],
        input_path=str(indir),
        output_path=str(outdir),
    )
    res = json.loads(res)
    assert res["environment"]["TESSERACT_INPUT_PATH"] == "/tesseract/input_data"
    assert res["environment"]["TESSERACT_OUTPUT_PATH"] == "/tesseract/output_data"
    assert res["volumes"].keys() == {str(outdir), str(indir), str(infile)}
    assert res["volumes"][str(indir)] == {
        "mode": "ro",
        "bind": "/tesseract/input_data",
    }

    with pytest.raises(ValueError):
        # test that input_path cannot be the same as output_path
        res, _ = engine.run_tesseract(
            "foobar",
            "apply",
            [f"@{infile}"],
            input_path=str(indir),
            output_path=str(indir),
        )

    with pytest.raises(ValueError):
        res, _ = engine.run_tesseract(
            "foobar",
            "apply",
            [f"@{infile}"],
            volumes=[f"{infile}:/some/path:ro"],
        )


def test_get_tesseract_images(mocked_docker):
    tesseract_images = engine.get_tesseract_images()
    assert len(tesseract_images) == 2


def test_get_tesseract_containers(mocked_docker):
    tesseract_containers = engine.get_tesseract_containers()
    assert len(tesseract_containers) == 1


def test_serve_tesseracts(mocked_docker):
    """Test multi-tesseract serve."""
    # Serve valid
    container_name_single_tesseract, _ = engine.serve("vectoradd")
    assert container_name_single_tesseract

    # Teardown valid
    engine.teardown(json.loads(container_name_single_tesseract)["name"])

    # Tear down invalid
    with pytest.raises(NotFound):
        engine.teardown("invalid_container_name")

    # Serve with gpus
    container_name_multi_tesseract, _ = engine.serve("vectoradd", gpus=["1", "3"])
    assert container_name_multi_tesseract

    # Teardown valid
    engine.teardown(json.loads(container_name_multi_tesseract)["name"])

    # Serve with memory
    container_name_with_memory, _ = engine.serve("vectoradd", memory="512m")
    assert container_name_with_memory

    # Teardown valid
    engine.teardown(json.loads(container_name_with_memory)["name"])


def test_serve_skip_health_check(mocked_docker, monkeypatch):
    """Test serving a tesseract with --skip-health-check."""
    health_called = False

    def health_get_spy(url, *args, **kwargs):
        nonlocal health_called
        if url.endswith("/health"):
            health_called = True
            return type("Response", (), {"status_code": 200, "json": lambda: {}})()
        raise NotImplementedError(f"Mocked get request to {url} not implemented")

    monkeypatch.setattr(engine.requests, "get", health_get_spy)

    res, _ = engine.serve("foobar", skip_health_check=True)
    assert res
    assert not health_called


def test_serve_memory(mocked_docker):
    """Test serving a tesseract with memory limit."""
    res, _ = engine.serve(
        "foobar",
        memory="2g",
    )

    res = json.loads(res)
    assert res["memory"] == "2g"


def test_serve_tesseract_volumes(mocked_docker, tmpdir):
    """Test running a tesseract with volumes."""
    # Test with a single volume
    res, _ = engine.serve(
        "foobar",
        volumes=[f"{tmpdir}:/path/in/container:ro"],
    )

    # Currently no good way to test return value of serve
    # since it returns a container name.
    res = json.loads(res)
    assert res["volumes"].keys() == {f"{tmpdir}"}
    assert res["volumes"][f"{tmpdir}"] == {
        "mode": "ro",
        "bind": "/path/in/container",
    }

    # Test with a named volume
    res, _ = engine.serve(
        "foobar",
        volumes=["my_named_volume:/path/in/container:ro"],
    )

    res = json.loads(res)
    assert res["volumes"].keys() == {"my_named_volume"}
    assert res["volumes"]["my_named_volume"] == {
        "mode": "ro",
        "bind": "/path/in/container",
    }

    with pytest.raises(RuntimeError):
        # Test with a volume that does not exist
        engine.serve(
            "foobar",
            volumes=["/non/existent/path:/path/in/container:ro"],
        )

    with pytest.raises(ValueError):
        # Test with a volume that has the same source path as another volume
        engine.serve(
            "foobar",
            volumes=[
                f"{tmpdir}:/path/in/container:ro",
                f"{tmpdir}:/path/in/container2:ro",
            ],
        )

    # Test running with input and output paths
    indir = Path(tmpdir / "input_path")
    indir.mkdir()
    outdir = Path(tmpdir) / "output1"
    outdir.mkdir()

    res, _ = engine.serve("foobar", input_path=str(indir), output_path=str(outdir))
    res = json.loads(res)
    assert res["volumes"].keys() == {str(indir), str(outdir)}
    assert res["volumes"][str(indir)] == {
        "mode": "ro",
        "bind": "/tesseract/input_data",
    }
    assert res["volumes"][str(outdir)] == {
        "mode": "rw",
        "bind": "/tesseract/output_data",
    }

    with pytest.raises(ValueError):
        # test that input_path cannot be the same as output_path
        engine.serve("foobar", input_path=str(indir), output_path=str(indir))


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


def test_log_streamer(tmpdir):
    # Test that LogStreamer can tail a file and capture lines as they are written
    from tesseract_core.sdk.logs import LogStreamer

    log_file = Path(tmpdir) / "test.log"
    captured = []

    streamer = LogStreamer(log_file, captured.append)
    streamer.start()

    # Write some lines with delays
    with open(log_file, "w") as f:
        for i in range(10):
            f.write(f"line {i}\n")
            f.flush()
            time.sleep(0.02)  # Small delay to let streamer pick up

    # Stop and drain
    streamer.stop()

    assert captured == [f"line {i}" for i in range(10)]


def test_log_streamer_waits_for_file(tmpdir):
    # Test that LogStreamer waits for the file to appear
    from tesseract_core.sdk.logs import LogStreamer

    log_file = Path(tmpdir) / "delayed.log"
    captured = []

    streamer = LogStreamer(log_file, captured.append)
    streamer.start()

    # File doesn't exist yet, streamer should be waiting
    time.sleep(0.1)

    # Now create the file
    with open(log_file, "w") as f:
        f.write("delayed line\n")
        f.flush()

    time.sleep(0.1)
    streamer.stop()

    assert captured == ["delayed line"]


def test_log_streamer_handles_trailing_content(tmpdir):
    # Test that LogStreamer handles content without trailing newline
    from tesseract_core.sdk.logs import LogStreamer

    log_file = Path(tmpdir) / "trailing.log"
    captured = []

    streamer = LogStreamer(log_file, captured.append)
    streamer.start()

    with open(log_file, "w") as f:
        f.write("line 1\n")
        f.write("no newline at end")
        f.flush()

    time.sleep(0.1)
    streamer.stop()

    assert captured == ["line 1", "no newline at end"]


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


def test_prepare_build_context_conda_no_env_file(tmp_path_factory):
    """Conda provider without an environment file does not error at staging."""
    src_dir = tmp_path_factory.mktemp("src")
    (src_dir / "tesseract_api.py").touch()
    build_dir = tmp_path_factory.mktemp("build")

    config = TesseractConfig(
        name="foobar",
        build_config=TesseractBuildConfig(requirements={"provider": "conda"}),
    )
    engine.prepare_build_context(src_dir, build_dir, config)
    assert (build_dir / "Dockerfile").exists()


@pytest.mark.parametrize(
    "line, expected",
    [
        ("./mylocaldep", ("./mylocaldep", "")),
        ("./mylocaldep[extra]", ("./mylocaldep", "[extra]")),
        ("../../pkg[a,b]", ("../../pkg", "[a,b]")),
        ("/abs/path", ("/abs/path", "")),
        ("./a[b] ", ("./a", "[b]")),
        ("file:///abs/path", ("/abs/path", "")),
        ("file:///abs/path[extra]", ("/abs/path", "[extra]")),
    ],
)
def test_split_local_dependency(line, expected):
    assert engine._split_local_dependency(line) == expected


def test_stage_local_dependency_file(tmp_path):
    """A local file dependency (e.g. a wheel) is copied, not tree-copied."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "mypkg-1.0-py3-none-any.whl").write_text("wheel contents")
    local_requirements = tmp_path / "local_requirements"
    local_requirements.mkdir()

    spec = engine._stage_local_dependency(
        "./mypkg-1.0-py3-none-any.whl", src_dir, local_requirements
    )
    staged = local_requirements / "mypkg-1.0-py3-none-any.whl"
    assert spec == "./local_requirements/mypkg-1.0-py3-none-any.whl"
    assert staged.is_file()
    assert staged.read_text() == "wheel contents"


def test_stage_local_dependency_name_collision(tmp_path):
    """Two dependencies resolving to the same basename get distinct staged names."""
    src_dir = tmp_path / "src"
    (src_dir / "a" / "mypkg").mkdir(parents=True)
    (src_dir / "b" / "mypkg").mkdir(parents=True)
    local_requirements = tmp_path / "local_requirements"
    local_requirements.mkdir()

    spec_a = engine._stage_local_dependency("./a/mypkg", src_dir, local_requirements)
    spec_b = engine._stage_local_dependency("./b/mypkg", src_dir, local_requirements)
    assert spec_a == "./local_requirements/mypkg"
    assert spec_b == "./local_requirements/mypkg_1"
    assert (local_requirements / "mypkg").is_dir()
    assert (local_requirements / "mypkg_1").is_dir()


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("/foo:/bar:ro", ["/foo", "/bar", "ro"]),
        ("/foo:/bar", ["/foo", "/bar"]),
        ("./foo:/bar:rw", ["./foo", "/bar", "rw"]),
        ("myvolume:/bar", ["myvolume", "/bar"]),
        ("C:\\Users\\foo:/bar:ro", ["C:\\Users\\foo", "/bar", "ro"]),
        ("C:\\Users\\foo:/bar", ["C:\\Users\\foo", "/bar"]),
        ("D:/data:/mnt/data:rw", ["D:/data", "/mnt/data", "rw"]),
    ],
)
def test_split_volume_spec(spec, expected):
    assert engine._split_volume_spec(spec) == expected


@pytest.mark.parametrize(
    "volume, expected",
    [
        ("/foo/bar", True),
        ("./foo", True),
        ("../foo", True),
        ("myvolume", False),
        ("C:\\Users\\foo", True),
        ("D:/data", True),
    ],
)
def test_is_local_volume(volume, expected):
    assert engine._is_local_volume(volume) == expected


@pytest.mark.parametrize(
    "within_range",
    [
        (-1, 100),
        (0, 65536),
        (100, 50),
    ],
)
def test_get_free_port_invalid_range(within_range):
    with pytest.raises(ValueError, match="Invalid port range"):
        engine.get_free_port(within_range=within_range)


def test_get_free_port_skips_port_in_use():
    """A port that is already bound is skipped in favor of a free one."""
    # Reserve two adjacent ports up front so we know both belong to the test,
    # then release one of them to act as the only free candidate.
    with (
        closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as occupied,
        closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as spare,
    ):
        occupied.bind(("127.0.0.1", 0))
        in_use = occupied.getsockname()[1]
        # Find an adjacent port we can also reserve, so the range holds exactly
        # the occupied port plus one known-free port.
        free = None
        for candidate in (in_use + 1, in_use - 1):
            if 0 <= candidate <= 65535:
                try:
                    spare.bind(("127.0.0.1", candidate))
                except OSError:
                    continue
                free = candidate
                break
        assert free is not None, "could not reserve an adjacent port"
        # Release the spare port so it becomes the only free port in the range.
        spare.close()

        within_range = (min(in_use, free), max(in_use, free) + 1)
        port = engine.get_free_port(within_range=within_range)

    assert port == free


def test_get_free_port_no_free_ports():
    """RuntimeError is raised when every port in the range is in use."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as occupied:
        occupied.bind(("127.0.0.1", 0))
        in_use = occupied.getsockname()[1]

        # Range containing only the single occupied port (end is exclusive).
        with pytest.raises(RuntimeError, match="No free ports found"):
            engine.get_free_port(within_range=(in_use, in_use + 1))


def test_get_free_port_all_excluded():
    """RuntimeError is raised when the only candidate port is excluded."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]

    with pytest.raises(RuntimeError, match="No free ports found"):
        engine.get_free_port(within_range=(free, free + 1), exclude=(free,))
