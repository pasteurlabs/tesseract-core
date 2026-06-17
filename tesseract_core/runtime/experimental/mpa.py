# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Metrics, Parameters, and Artifacts (MPA) logging API.

These functions log to the run context established by
:func:`tesseract_core.runtime.mpa.start_run`, which is entered automatically
when a Tesseract endpoint is served.

.. note::

    These are experimental and the API may change in future releases.
"""

from typing import Any

from tesseract_core.runtime.mpa import _get_current_backend


def log_parameter(key: str, value: Any) -> None:
    """Log a parameter to the current run context."""
    _get_current_backend().log_parameter(key, value)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a metric to the current run context."""
    _get_current_backend().log_metric(key, value, step)


def log_artifact(local_path: str) -> None:
    """Log an artifact to the current run context."""
    _get_current_backend().log_artifact(local_path)
