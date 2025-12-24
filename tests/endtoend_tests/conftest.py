# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for end-to-end tests.

Provides fixtures and hooks for parameterized regression testing.
"""

from pathlib import Path

import pytest
from common import build_tesseract, image_exists

from tesseract_core.sdk.tesseract import Tesseract

# Get examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
UNIT_TESSERACTS = [
    tr.stem for tr in EXAMPLES_DIR.glob("*/") if not tr.stem.startswith("_")
]


def discover_regress_tests(example_path: Path) -> list[Path]:
    """Discover all regress_*.json test files for an example."""
    test_cases_dir = example_path / "test_cases"
    if not test_cases_dir.exists():
        return []
    return sorted(test_cases_dir.glob("regress_*.json"))


# Build parametrization for all (tesseract, test_file) combinations
_test_params = []
_test_ids = []

for tesseract_name in UNIT_TESSERACTS:
    tesseract_path = EXAMPLES_DIR / tesseract_name
    test_files = discover_regress_tests(tesseract_path)

    if test_files:
        for test_file in test_files:
            _test_params.append((tesseract_path, test_file))
            _test_ids.append(f"{tesseract_name}-{test_file.name}")
    else:
        # No test files for this tesseract - add skip placeholder
        _test_params.append((tesseract_path, None))
        _test_ids.append(f"{tesseract_name}-no_tests")


@pytest.fixture(scope="session", params=_test_params, ids=_test_ids)
def unit_tesseract_path_and_test_file(request):
    """Override parent fixture to parametrize by (tesseract, test_file) pairs."""
    return request.param


@pytest.fixture(scope="session")
def unit_tesseract_path(unit_tesseract_path_and_test_file):
    """Extract just the tesseract path from the parametrized pair."""
    return unit_tesseract_path_and_test_file[0]


@pytest.fixture(scope="session")
def regress_test_file(unit_tesseract_path_and_test_file):
    """Extract just the test file from the parametrized pair."""
    return unit_tesseract_path_and_test_file[1]


_tesseract_cache = {}  # Cache instances by tesseract path


@pytest.fixture(scope="function")
def tesseract_instance(
    docker_client, dummy_image_name, unit_tesseract_path, docker_cleanup
):
    """Build and serve a Tesseract instance, reused for all regression tests of same tesseract.

    Uses function scope with manual caching to ensure we build once per tesseract,
    even though it's called for every test.
    """
    # Check cache first
    tesseract_key = str(unit_tesseract_path)

    if tesseract_key in _tesseract_cache:
        cached_instance = _tesseract_cache[tesseract_key]
        if cached_instance is None:
            # This tesseract has no tests
            yield None
            return
        # Verify the cached instance is still valid (container still running)
        try:
            cached_instance.health()  # Ping to verify server is alive
            yield cached_instance
            return
        except Exception:
            # Server is dead, remove from cache and rebuild
            _tesseract_cache.pop(tesseract_key, None)

    # Build image once per tesseract
    img_name = build_tesseract(
        docker_client,
        unit_tesseract_path,
        dummy_image_name,
        tag="sometag",
    )
    assert image_exists(docker_client, img_name)
    docker_cleanup["images"].append(img_name)

    # Discover test files for this tesseract
    test_files = discover_regress_tests(unit_tesseract_path)

    if not test_files:
        # Skip if no test files - cache None
        _tesseract_cache[tesseract_key] = None
        yield None
        return

    # Serve once, reuse for all regression tests
    # Don't use context manager since we want to cache and reuse
    tess = Tesseract.from_image(img_name)
    tess.__enter__()  # Start serving
    docker_cleanup["containers"].append(tess._serve_context["container_name"])
    _tesseract_cache[tesseract_key] = tess
    yield tess
    # Don't call __exit__ yet - will be reused by other tests
    # Cleanup happens via docker_cleanup fixture at session end
