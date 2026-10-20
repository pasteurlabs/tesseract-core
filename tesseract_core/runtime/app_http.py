# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instantiated version of the FastAPI app."""

from tesseract_core.runtime.core import get_tesseract_api, use_utf8_stdio
from tesseract_core.runtime.serve import create_rest_api

# Workers spawned by `serve --num-workers` import this without going through the CLI.
use_utf8_stdio()

tesseract_api = get_tesseract_api()
app = create_rest_api(tesseract_api)
