# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for ``python -m tesseract_core``.

Equivalent to the ``tesseract`` console script, but reachable via an interpreter
path alone. Useful when its scripts directory is not on ``PATH``, and when
``tesseract`` on this machine is the OCR engine of the same name.
"""

from tesseract_core.sdk.cli import entrypoint

if __name__ == "__main__":
    entrypoint()
