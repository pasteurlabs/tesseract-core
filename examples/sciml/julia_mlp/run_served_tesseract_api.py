# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Simple development file used to show how to make requests to the julia_flux_mlp tesseract
# assumes the tesseract is located at: http://127.0.0.1:8000

from tesseract_core import Tesseract

with Tesseract.from_image(image="julia_flux_mlp") as julia_flux_mlp:
    data = julia_flux_mlp.apply(inputs={"n_epochs": 10})
state = data["state"]
print(data["metrics"])
