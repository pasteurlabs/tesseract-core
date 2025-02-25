# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path

# Recursive import of the whole package to ensure all deps are present
try:
    for path in Path(__file__).parent.glob("**/*.py"):
        if path.stem == "__init__":
            continue
        if path.stem.startswith("app_"):
            continue
        if not (path.parent / "__init__.py").exists():
            continue

        package_path = ".".join(
            path.relative_to(Path(__file__).parent).with_suffix("").parts
        )
        importlib.import_module(f".{package_path}", __package__)
except ModuleNotFoundError as e:
    print(
        f"Failed to import {path}: {e}\n\n",
        "You are likely missing dependencies. "
        "Try running the following command to install runtime dependencies:\n\n"
        "   $ pip install tesseract-core[runtime]\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Import public API
from .schema_types import (
    Array,
    Differentiable,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    ShapeDType,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

__all__ = [
    "Array",
    "Differentiable",
    "Float16",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "ShapeDType",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
]
