#!/bin/bash

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

uv venv /python-env
source /python-env/bin/activate

# Collect dependencies
TESSERACT_DEPS=$(find ./local_requirements/ -mindepth 1 -maxdepth 1 2>/dev/null || true)

# Append requirements file
TESSERACT_DEPS+=" -r tesseract_requirements.txt"

# Install dependencies
uv -v pip install --compile-bytecode $TESSERACT_DEPS

# HACK: If `tesseract_core` is part of tesseract_requirements.txt, it may install an incompatible version
# of the runtime from PyPI. We remove the runtime folder and install the local version instead.
runtime_path=$(python -c "import tesseract_core; print(tesseract_core.__file__.replace('__init__.py', ''))" || true)
if [ -d "$runtime_path" ]; then
    rm -rf "$runtime_path"/runtime
fi

uv -v pip install --compile-bytecode ./tesseract_runtime

# Install pip itself into the virtual environment for use by any custom build steps
uv pip install pip
