# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling and tracing functionality."""

import json
import subprocess

import numpy as np
import pytest

from tesseract_core import Tesseract


@pytest.fixture(autouse=True)
def use_dummy_tesseract(dummy_tesseract):
    """Use dummy tesseract for all tests."""
    yield


class TestProfilerCLI:
    """Tests for profiling via tesseract-runtime CLI."""

    def test_profiling_disabled_by_default(self, dummy_tesseract_package):
        """Verify profiling output is not present when profiling is disabled."""
        api_path = dummy_tesseract_package / "tesseract_api.py"
        payload = json.dumps({"inputs": {"a": [1.0, 2.0], "b": [3.0, 4.0], "s": 1.0}})

        result = subprocess.run(
            ["tesseract-runtime", "apply", payload],
            capture_output=True,
            text=True,
            env={
                **dict(__import__("os").environ),
                "TESSERACT_API_PATH": str(api_path),
            },
        )

        assert result.returncode == 0, result.stderr
        assert "--- Profiling Statistics ---" not in result.stderr
        assert "--- Profiling Statistics ---" not in result.stdout

    def test_profiling_enabled_via_env(self, dummy_tesseract_package):
        """Verify profiling output is present when TESSERACT_PROFILING=1."""
        api_path = dummy_tesseract_package / "tesseract_api.py"
        payload = json.dumps({"inputs": {"a": [1.0, 2.0], "b": [3.0, 4.0], "s": 1.0}})

        result = subprocess.run(
            ["tesseract-runtime", "apply", payload],
            capture_output=True,
            text=True,
            env={
                **dict(__import__("os").environ),
                "TESSERACT_API_PATH": str(api_path),
                "TESSERACT_PROFILING": "1",
            },
        )

        assert result.returncode == 0, result.stderr
        # Profiling output goes to stderr (via stdout redirect to log)
        combined_output = result.stdout + result.stderr
        assert "--- Profiling Statistics ---" in combined_output
        # Check for profiling report sections
        assert "By Cumulative Time" in combined_output
        assert "By Total Time" in combined_output

    def test_profiling_output_contains_function_stats(self, dummy_tesseract_package):
        """Verify profiling output contains actual function statistics."""
        api_path = dummy_tesseract_package / "tesseract_api.py"
        payload = json.dumps({"inputs": {"a": [1.0, 2.0], "b": [3.0, 4.0], "s": 1.0}})

        result = subprocess.run(
            ["tesseract-runtime", "apply", payload],
            capture_output=True,
            text=True,
            env={
                **dict(__import__("os").environ),
                "TESSERACT_API_PATH": str(api_path),
                "TESSERACT_PROFILING": "1",
            },
        )

        assert result.returncode == 0, result.stderr
        combined_output = result.stdout + result.stderr
        # pstats output typically contains these column headers
        assert "ncalls" in combined_output or "tottime" in combined_output


class TestProfilerSDK:
    """Tests for profiling via Tesseract.from_tesseract_api."""

    def test_profiling_disabled_by_default(self, dummy_tesseract_package, capsys):
        """Verify profiling is disabled by default in SDK."""
        tess = Tesseract.from_tesseract_api(
            dummy_tesseract_package / "tesseract_api.py"
        )

        _ = tess.apply(
            inputs={
                "a": np.array([1.0, 2.0], dtype=np.float32),
                "b": np.array([3.0, 4.0], dtype=np.float32),
                "s": 1.0,
            }
        )

        captured = capsys.readouterr()
        assert "--- Profiling Statistics ---" not in captured.out
        assert "--- Profiling Statistics ---" not in captured.err

    def test_profiling_enabled_via_runtime_config(
        self, dummy_tesseract_package, capsys
    ):
        """Verify profiling can be enabled via runtime_config."""
        tess = Tesseract.from_tesseract_api(
            dummy_tesseract_package / "tesseract_api.py",
            runtime_config={"profiling": True},
        )

        _ = tess.apply(
            inputs={
                "a": np.array([1.0, 2.0], dtype=np.float32),
                "b": np.array([3.0, 4.0], dtype=np.float32),
                "s": 1.0,
            }
        )

        captured = capsys.readouterr()
        combined_output = captured.out + captured.err
        assert "--- Profiling Statistics ---" in combined_output


class TestTracingCLI:
    """Tests for tracing via tesseract-runtime CLI."""

    def test_tracing_disabled_by_default(self, dummy_tesseract_package):
        """Verify tracing output is not present when tracing is disabled."""
        api_path = dummy_tesseract_package / "tesseract_api.py"
        payload = json.dumps({"inputs": {"a": [1.0, 2.0], "b": [3.0, 4.0], "s": 1.0}})

        result = subprocess.run(
            ["tesseract-runtime", "apply", payload],
            capture_output=True,
            text=True,
            env={
                **dict(__import__("os").environ),
                "TESSERACT_API_PATH": str(api_path),
            },
        )

        assert result.returncode == 0, result.stderr
        # Tracing output would contain DEBUG level messages
        assert "DEBUG" not in result.stderr

    def test_tracing_enabled_via_env(self, dummy_tesseract_package):
        """Verify tracing output is present when TESSERACT_TRACING=1."""
        api_path = dummy_tesseract_package / "tesseract_api.py"
        payload = json.dumps({"inputs": {"a": [1.0, 2.0], "b": [3.0, 4.0], "s": 1.0}})

        result = subprocess.run(
            ["tesseract-runtime", "apply", payload],
            capture_output=True,
            text=True,
            env={
                **dict(__import__("os").environ),
                "TESSERACT_API_PATH": str(api_path),
                "TESSERACT_TRACING": "1",
            },
        )

        assert result.returncode == 0, result.stderr
        # When tracing is enabled, DEBUG messages should appear
        # The runtime logger outputs to stdout
        combined_output = result.stdout + result.stderr
        assert "DEBUG" in combined_output or "tesseract_runtime" in combined_output


class TestTracingSDK:
    """Tests for tracing via Tesseract.from_tesseract_api."""

    def test_tracing_enabled_via_runtime_config(self, dummy_tesseract_package, capsys):
        """Verify tracing can be enabled via runtime_config."""
        tess = Tesseract.from_tesseract_api(
            dummy_tesseract_package / "tesseract_api.py",
            runtime_config={"tracing": True},
        )

        _ = tess.apply(
            inputs={
                "a": np.array([1.0, 2.0], dtype=np.float32),
                "b": np.array([3.0, 4.0], dtype=np.float32),
                "s": 1.0,
            }
        )

        captured = capsys.readouterr()
        combined_output = captured.out + captured.err
        # When tracing is enabled, we should see DEBUG level output
        assert "DEBUG" in combined_output or "tesseract_runtime" in combined_output
