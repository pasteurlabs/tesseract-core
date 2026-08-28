# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for building Tesseract images."""

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from common import build_tesseract, image_exists

from tesseract_core.sdk.cli import AVAILABLE_RECIPES, app

tested_images = ("ubuntu:24.04",)

build_matrix = [
    *[(r, None) for r in AVAILABLE_RECIPES],
    *[(None, img) for img in tested_images],
]


@pytest.mark.parametrize("recipe,base_image", build_matrix)
def test_build_from_init_endtoend(
    cli_runner,
    docker_client,
    docker_cleanup,
    dummy_image_name,
    tmp_path,
    recipe,
    base_image,
):
    """Test that a trivial (empty) Tesseract image can be built from init."""
    init_args = ["init", "--target-dir", str(tmp_path), "--name", dummy_image_name]
    if recipe:
        init_args.extend(["--recipe", recipe])

    result = cli_runner.invoke(app, init_args, catch_exceptions=False)
    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "tesseract_api.py").exists()
    with open(tmp_path / "tesseract_config.yaml") as config_yaml:
        assert yaml.safe_load(config_yaml)["name"] == dummy_image_name

    config_override = {}
    if base_image is not None:
        config_override["build_config.base_image"] = base_image

    image_name = build_tesseract(
        docker_client,
        tmp_path,
        dummy_image_name,
        config_override=config_override,
    )

    docker_cleanup["images"].append(image_name)
    assert image_exists(docker_client, image_name)

    # Test that the image can be run and that --help is forwarded correctly
    result = cli_runner.invoke(
        app,
        [
            "run",
            image_name,
            "apply",
            "--help",
        ],
        catch_exceptions=False,
    )
    assert f"Usage: tesseract run {image_name} apply" in result.stderr


def test_build_with_tag(
    cli_runner,
    docker_client,
    docker_cleanup,
    dummy_image_name,
    tmp_path,
):
    """Test that a Tesseract image can be built with a custom tag."""
    init_args = ["init", "--target-dir", str(tmp_path), "--name", dummy_image_name]
    result = cli_runner.invoke(app, init_args, catch_exceptions=False)
    assert result.exit_code == 0, result.stderr

    image_name = build_tesseract(
        docker_client,
        tmp_path,
        dummy_image_name,
        tag="foo",
    )

    docker_cleanup["images"].append(image_name)
    assert image_exists(docker_client, image_name)
    assert image_name.endswith(":foo")


@pytest.mark.parametrize("skip_checks", [True, False])
def test_build_generate_only(cli_runner, dummy_tesseract_location, skip_checks):
    """Test output of build with --generate_only flag."""
    build_res = cli_runner.invoke(
        app,
        [
            "build",
            str(dummy_tesseract_location),
            "--generate-only",
            *(
                ("--config-override=build_config.skip_checks=True",)
                if skip_checks
                else ()
            ),
        ],
        # Ensure that the output is not truncated
        env={"COLUMNS": "1000"},
        catch_exceptions=False,
    )
    assert build_res.exit_code == 0, build_res.stderr
    # Check that stdout contains build command
    command = "buildx build"
    assert command in build_res.stderr

    build_dir = Path(build_res.stdout.strip())
    assert build_dir.exists()
    dockerfile_path = build_dir / "Dockerfile"
    assert dockerfile_path.exists()

    with open(build_dir / "Dockerfile") as f:
        docker_file_contents = f.read()
        if skip_checks:
            assert "tesseract-runtime check" not in docker_file_contents
        else:
            assert "tesseract-runtime check" in docker_file_contents


def test_tarball_install(cli_runner, dummy_tesseract_package, docker_cleanup):
    tesseract_api = dedent(
        """
    import cowsay
    from pydantic import BaseModel

    class InputSchema(BaseModel):
        message: str = "Hello, Tesseractor!"

    class OutputSchema(BaseModel):
        out: str

    def apply(inputs: InputSchema) -> OutputSchema:
        return OutputSchema(out=cowsay.get_output_string("cow", inputs.message))
    """
    )

    tesseract_requirements = "./cowsay-6.1-py3-none-any.whl"

    subprocess.run(
        ["pip", "download", "cowsay==6.1", "-d", str(dummy_tesseract_package)]
    )
    with open(dummy_tesseract_package / "tesseract_api.py", "w") as f:
        f.write(tesseract_api)
    with open(dummy_tesseract_package / "tesseract_requirements.txt", "w") as f:
        f.write(tesseract_requirements)

    result = cli_runner.invoke(
        app,
        ["--loglevel", "debug", "build", str(dummy_tesseract_package)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr

    img_tag = json.loads(result.stdout)[0]
    docker_cleanup["images"].append(img_tag)


def test_build_extra_index_url_with_local_dep(
    cli_runner, dummy_tesseract_package, docker_cleanup
):
    """A supplementary index (--extra-index-url) must not shadow PyPI.

    Regression test for #651: an ``--index-url`` pins resolution to a single
    index that does not host build backends, so a local path dependency using a
    non-setuptools backend (here hatchling) fails to build. ``--extra-index-url``
    supplements PyPI instead of replacing it, so the backend still resolves.
    """
    local_dep = dummy_tesseract_package / "mylib"
    (local_dep / "mylib").mkdir(parents=True)
    (local_dep / "mylib" / "__init__.py").touch()
    (local_dep / "pyproject.toml").write_text(
        dedent(
            """
            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [project]
            name = "mylib"
            version = "0.1.0"
            """
        )
    )

    (dummy_tesseract_package / "tesseract_requirements.txt").write_text(
        dedent(
            """
            --extra-index-url https://download.pytorch.org/whl/cpu
            ./mylib
            """
        )
    )

    result = cli_runner.invoke(
        app,
        ["build", str(dummy_tesseract_package)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr
    docker_cleanup["images"].append(json.loads(result.stdout)[0])


def test_build_env_and_host_credential_with_secret(
    cli_runner, dummy_tesseract_package, docker_cleanup
):
    """build_env + an authenticated host credential build, with no credential leak.

    Exercises #676 (build_env reaching the resolver) and #675 (a host credential
    whose token is supplied out-of-band via ``--secret``, mounted at build time,
    and assembled into netrc + git-credential entries). PyTorch's CPU host is used
    as the credentialed host; with the default first-index strategy numpy still
    resolves from PyPI, so the authenticated host is never actually queried and
    the build stays deterministic while the secret-mount + credential-setup code
    path runs. An invalid build_env value would make uv error, so a successful
    build also proves build_env reaches the resolver. Afterwards we assert the
    secret value never lands in the image config or history.
    """
    secret_value = "s3cr3t-token-value"

    config_path = dummy_tesseract_package / "tesseract_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    build_config = config.setdefault("build_config", {})
    build_config["build_env"] = {"UV_INDEX_STRATEGY": "first-index"}
    build_config["host_credentials"] = [
        {"host": "download.pytorch.org", "secret_id": "host_token"},
    ]
    config_path.write_text(yaml.dump(config))

    result = cli_runner.invoke(
        app,
        [
            "build",
            str(dummy_tesseract_package),
            "--secret",
            "id=host_token,env=HOST_TOKEN",
        ],
        env={"HOST_TOKEN": secret_value},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr
    image_tag = json.loads(result.stdout)[0]
    docker_cleanup["images"].append(image_tag)

    # The secret must not be recoverable from image metadata or layer history
    # (covers both the netrc and git-credentials files, which live only in the
    # build stage).
    inspect = subprocess.run(
        ["docker", "inspect", image_tag], capture_output=True, text=True
    )
    assert secret_value not in inspect.stdout
    history = subprocess.run(
        ["docker", "history", "--no-trunc", image_tag],
        capture_output=True,
        text=True,
    )
    assert secret_value not in history.stdout
    # The credentials must also not survive into the final image filesystem.
    grep = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image_tag,
            "-c",
            "cat ~/.netrc ~/.git-credentials 2>/dev/null || true",
        ],
        capture_output=True,
        text=True,
    )
    assert secret_value not in grep.stdout


def test_metadata_label(built_image_name):
    """Test that metadata from tesseract_config.yaml is stored as a Docker label."""
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            '{{ index .Config.Labels "ai.pasteurlabs.tesseract.metadata" }}',
            built_image_name,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    label_data = json.loads(result.stdout.strip())
    assert label_data == {"tags": ["ml", "physics"], "nested": {"key": "value"}}
