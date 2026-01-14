# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for all unit Tesseracts.

Each unit Tesseract is tested by building a Docker image from it and then
running the Tesseract CLI + HTTP interface to test its functionality.

Add test cases for specific unit Tesseracts to the TEST_CASES dictionary.
"""

import base64
import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import requests
from common import build_tesseract, image_exists

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def json_normalize(obj: str):
    """Normalize JSON str for comparison."""
    return json.dumps(json.loads(obj), separators=(",", ":"))


def assert_contains_array_allclose(
    output: dict | list,
    array_expected: npt.ArrayLike,
    rtol=1e-4,
    atol=1e-4,
):
    """Check the output pytree for an array close to array_expected."""

    def _get_array_leaves(tree):
        if isinstance(tree, dict) and tree.get("object_type") == "array":
            shape = tree["shape"]
            dtype = tree["dtype"]
            data = tree["data"]
            if data["encoding"] == "base64":
                buffer = base64.b64decode(data["buffer"])
                array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
            else:
                raise NotImplementedError("only base64 encoding supported")
            yield array
        elif isinstance(tree, dict):
            for val in tree.values():
                yield from _get_array_leaves(val)
        elif isinstance(tree, list):
            for val in tree:
                yield from _get_array_leaves(val)
        else:
            raise AssertionError(f"Got unexpected type {type(tree)}")

    output_arrays = list(_get_array_leaves(output))
    expected = np.asarray(array_expected)
    for array in output_arrays:
        if np.allclose(array, expected, rtol=rtol, atol=atol):
            return

    raise AssertionError(
        f"Expected array not found in output.\nExpected: {expected}\nFound arrays: {output_arrays}"
    )


@dataclass
class Config:
    test_with_random_inputs: bool = False
    check_gradients: bool = False
    volume_mounts: list[str] = None
    input_path: str = None
    output_path: str = None


# Add config and test cases for specific unit Tesseracts here
TEST_CASES = {
    "empty": Config(test_with_random_inputs=True),
    "py310": Config(test_with_random_inputs=True),
    "helloworld": Config(test_with_random_inputs=True),
    "pip_custom_step": Config(test_with_random_inputs=True),
    "pyvista-arm64": Config(test_with_random_inputs=True),
    "localpackage": Config(test_with_random_inputs=True),
    "vectoradd": Config(test_with_random_inputs=True),
    "vectoradd_jax": Config(test_with_random_inputs=True, check_gradients=True),
    "vectoradd_torch": Config(test_with_random_inputs=True),
    "univariate": Config(test_with_random_inputs=True, check_gradients=True),
    "package_data": Config(test_with_random_inputs=True),
    "cuda": Config(test_with_random_inputs=True),
    "meshstats": Config(check_gradients=True),
    "dataloader": Config(
        check_gradients=True, volume_mounts=["testdata:/tesseract/input_data:ro"]
    ),
    "conda": Config(),
    "required_files": Config(input_path="input"),
    "filereference": Config(input_path="test_cases/testdata", output_path="output"),
    "metrics": Config(test_with_random_inputs=True),
    "qp_solve": Config(),
    "tesseractreference": Config(),  # Can't test requests standalone; needs target Tesseract. Covered in separate test.
    "userhandling": Config(),
}


@pytest.fixture
def unit_tesseract_config(unit_tesseract_names, unit_tesseract_path):
    for tesseract in TEST_CASES:
        if tesseract not in unit_tesseract_names:
            raise ValueError(
                f"A test case in TEST_CASES refers to a nonexistent Tesseract {tesseract}."
            )

    if unit_tesseract_path.name not in TEST_CASES:
        raise ValueError(
            f"No test case found for Tesseract {unit_tesseract_path.name}."
        )

    return TEST_CASES[unit_tesseract_path.name]


def print_debug_info(result):
    """Print debug info from result of a CLI command if it failed."""
    if result.exit_code == 0:
        return
    print(result.output)
    if result.exc_info:
        traceback.print_exception(*result.exc_info)


def fix_fake_arrays(fakedata, seed=42):
    is_array = (
        lambda x: isinstance(x, dict) and "shape" in x and "dtype" in x and "data" in x
    )
    rng = np.random.RandomState(seed)

    def _walk(data):
        if is_array(data):
            # Use broadcasting to minimize surface for shape errors
            data["shape"] = (1,) * len(data["shape"])

            new_data = rng.uniform(0, 100, data["shape"]).astype(data["dtype"])

            if data["data"]["encoding"] == "base64":
                data["data"]["buffer"] = base64.b64encode(new_data.tobytes()).decode()
            elif data["data"]["encoding"] == "json":
                data["data"]["buffer"] = new_data.flatten().tolist()
            elif data["data"]["encoding"] == "binref":
                # FIXME: overriding with base64 to not have to create files
                # in the tesseract
                data["data"]["encoding"] = "base64"
                data["data"]["buffer"] = base64.b64encode(new_data.tobytes()).decode()
        elif isinstance(data, dict):
            for key, value in data.items():
                data[key] = _walk(value)
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                data[idx] = _walk(value)

        return data

    return _walk(fakedata)


def example_from_json_schema(schema):
    """Generate a random example JSON object from a JSON schema."""
    import jsf

    faker = jsf.JSF(schema)
    payload = faker.generate()
    payload = fix_fake_arrays(payload)
    return payload


def test_unit_tesseract_endtoend(
    cli_runner,
    docker_client,
    dummy_image_name,
    unit_tesseract_path,
    unit_tesseract_config,
    free_port,
    docker_cleanup,
):
    """Test that unit Tesseract images can be built and used to serve REST API."""
    from tesseract_core.sdk.cli import app

    # Stage 1: Build
    img_name = build_tesseract(
        docker_client,
        unit_tesseract_path,
        dummy_image_name,
        tag="sometag",
    )
    assert image_exists(docker_client, img_name)
    docker_cleanup["images"].append(img_name)

    # Stage 2: Test CLI usage
    result = cli_runner.invoke(
        app,
        [
            "run",
            img_name,
            "openapi-schema",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    openapi_schema = json.loads(result.stdout)

    def _input_schema_from_openapi(openapi_schema):
        input_schema = openapi_schema["components"]["schemas"]["ApplyInputSchema"]
        # For some reason, jsf can't handle #/components/schemas/<x> references,
        # so we convert them to #$defs/<x>
        input_schema.update({"$defs": openapi_schema["components"]["schemas"]})
        input_schema["$defs"].pop("ApplyInputSchema", None)
        input_schema = json.loads(
            json.dumps(input_schema).replace("components/schemas", "$defs")
        )
        return input_schema

    input_schema = _input_schema_from_openapi(openapi_schema)

    mount_args, io_args = [], []

    if unit_tesseract_config.volume_mounts:
        for mnt in unit_tesseract_config.volume_mounts:
            # Assume that the mount is relative to the Tesseract path
            local_path, *other = mnt.split(":")
            local_path = Path(local_path)
            if not local_path.is_absolute():
                local_path = unit_tesseract_path / local_path
            mnt = ":".join([str(local_path), *other])
            mount_args.extend(["--volume", mnt])

    if unit_tesseract_config.input_path:
        io_args.extend(
            [
                "--input-path",
                str(unit_tesseract_path / unit_tesseract_config.input_path),
            ]
        )
    if unit_tesseract_config.output_path:
        io_args.extend(
            [
                "--output-path",
                str(unit_tesseract_path / unit_tesseract_config.output_path),
            ]
        )

    if unit_tesseract_config.test_with_random_inputs:
        random_input = example_from_json_schema(input_schema)

        result = cli_runner.invoke(
            app,
            [
                "run",
                img_name,
                *mount_args,
                "apply",
                json.dumps(random_input),
                *io_args,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    test_cases_dir = unit_tesseract_path / "test_cases"
    if test_cases_dir.exists() and test_cases_dir.is_dir():
        test_files = sorted(test_cases_dir.glob("*.json"))
        for test_file_path in test_files:
            test_file = test_file_path.relative_to(unit_tesseract_path)
            print(f"Running regress test: {test_file}")

            args = [
                "run",
                img_name,
                *mount_args,
                "regress",
                f"@{test_file_path}",
                *io_args,
            ]

            result = cli_runner.invoke(app, args, env={"TERM": "dumb"})
            print_debug_info(result)
            assert result.exit_code == 0, result.exception

            # Parse regress response
            regress_result = json.loads(result.stdout)
            assert regress_result["status"] == "passed", (
                f"Regress test failed for {test_file}:\n"
                f"  Endpoint: {regress_result['endpoint']}\n"
                f"  Message: {regress_result['message']}"
            )

    # check-gradients is a CLI command not a true endpoint
    # as such the regress endpoint cannot access it directly.
    # Therefore, we only test with cli_runner (stage 2)
    if unit_tesseract_config.check_gradients:
        checkgradients_input = (
            unit_tesseract_path / "test_cases_inputs/example_checkgradients_input.json"
        )
        if checkgradients_input.exists():
            print(f"Running check-gradients test: {checkgradients_input}")

            with open(checkgradients_input) as f:
                payload = json.load(f)

            args = [
                "run",
                img_name,
                *mount_args,
                "check-gradients",
                json.dumps(payload),
            ]

            result = cli_runner.invoke(app, args, env={"TERM": "dumb"})
            print_debug_info(result)
            assert result.exit_code == 0, result.exception
            # Verify check-gradients passed (check stdout or stderr for success indicators)
            output_text = result.stdout + result.stderr
            assert "passed" in output_text.lower() or "✅" in output_text

    # Stage 3: Test HTTP server
    run_res = cli_runner.invoke(
        app,
        [
            "serve",
            img_name,
            "-p",
            free_port,
            "--debug",
            *mount_args,
            *io_args,
        ],
        catch_exceptions=False,
    )

    assert run_res.exit_code == 0, run_res.stderr
    assert run_res.stdout

    serve_meta = json.loads(run_res.stdout)
    container_name = serve_meta["container_name"]
    docker_cleanup["containers"].append(container_name)

    # Now test server (send requests and validate outputs)
    response = requests.get(f"http://localhost:{free_port}/openapi.json")
    assert response.status_code == 200
    openapi_schema = response.json()
    input_schema = _input_schema_from_openapi(openapi_schema)

    if unit_tesseract_config.test_with_random_inputs:
        payload_from_schema = example_from_json_schema(input_schema)
        response = requests.post(
            f"http://localhost:{free_port}/apply", json=payload_from_schema
        )
        assert response.status_code == 200, response.text
