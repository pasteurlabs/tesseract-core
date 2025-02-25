# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instantiated version of the FastAPI app."""

from tesseract_core.runtime.core import get_tesseract_api
from tesseract_core.runtime.serve import create_rest_api

tesseract_api = get_tesseract_api()
app = create_rest_api(tesseract_api)
