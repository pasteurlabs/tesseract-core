# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for ``python -m tesseract_core.runtime``.

Equivalent to the ``tesseract-runtime`` console script, but reachable via an
interpreter path alone, without knowing where its scripts directory is.
"""

from tesseract_core.runtime.cli import main

if __name__ == "__main__":
    main()
