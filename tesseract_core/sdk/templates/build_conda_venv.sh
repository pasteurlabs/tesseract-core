#!/bin/bash

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

conda env create --file tesseract_environment.yaml -p /python-env --quiet
conda run -p /python-env pip install ./tesseract_runtime
