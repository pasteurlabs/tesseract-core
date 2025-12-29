# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for all unit Tesseracts.

Each regress_*.json file becomes a separate pytest test via parametrization.
Test discovery and Tesseract instance management is handled by conftest.py.
"""

import json

import pytest


def test_regress_case(tesseract_instance, regress_test_file):
    """Test a single regression case.

    This test is parametrized by pytest_generate_tests in conftest.py to create
    one test per regress_*.json file in the tesseract's test_cases/ directory.

    The Tesseract instance is built once per tesseract and reused for all tests.
    """
    if tesseract_instance is None or regress_test_file is None:
        pytest.skip("No regress test files found for this tesseract")

    test_spec = json.loads(regress_test_file.read_text())
    tesseract_instance.regress(test_spec)  # Raises AssertionError on failure
