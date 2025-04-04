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
pip install ./tesseract_runtime
