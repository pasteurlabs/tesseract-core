#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

if [ -f tesseract_environment.yml ]; then
    conda env create --file tesseract_environment.yml -p /python-env
    conda run -p /python-env pip install ./tesseract_runtime
fi
