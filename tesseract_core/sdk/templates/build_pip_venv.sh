#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

python3 -m venv /python-env

# Activate virtual environment
source /python-env/bin/activate

# Collect dependencies
TESSERACT_DEPS=$(find ./local_requirements/ -mindepth 1 -maxdepth 1 2>/dev/null || true)

# Append requirements file
TESSERACT_DEPS+=" -r tesseract_requirements.txt"

# Install dependencies
pip install $TESSERACT_DEPS

# HACK: If `tesseract_core` is part of tesseract_requirements.txt, it may install an incompatible version
# of the runtime from PyPI. We remove the runtime folder and install the local version instead.
runtime_path=$(python -c "import tesseract_core; print(tesseract_core.__file__.replace('__init__.py', ''))" || true)
if [ -d "$runtime_path" ]; then
    rm -rf "$runtime_path"/runtime
fi

pip install ./tesseract_runtime
