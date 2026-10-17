# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instantiated version of the FastAPI app."""

import sys

from tesseract_core.runtime.core import get_tesseract_api
from tesseract_core.runtime.serve import create_rest_api

# Every serving process imports this, and its output is read back as UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

tesseract_api = get_tesseract_api()
app = create_rest_api(tesseract_api)
