#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

if [[ -f tesseract_requirements.txt ]]; then
    python3 -m venv /python-env

    # Collect dependencies
    TESSERACT_DEPS=$(find ./local_requirements/ -mindepth 1 -maxdepth 1 2>/dev/null || true)

    # Append requirements file
    TESSERACT_DEPS+=" -r tesseract_requirements.txt"

    # Activate virtual environment
    source /python-env/bin/activate

    # Install dependencies
    pip install ./tesseract_runtime $TESSERACT_DEPS
fi
