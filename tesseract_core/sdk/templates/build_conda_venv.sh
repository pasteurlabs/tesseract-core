#!/bin/bash

# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

# Set up host credentials (netrc + git-credentials) for authenticated conda
# channels, pip indices, and git+https dependencies. No-op if none declared.
source setup_host_credentials.sh

conda env create --file tesseract_environment.yaml -p /python-env --quiet
conda run -p /python-env pip install ./tesseract_runtime
