import yaml
import torch
from functools import partial
from pathlib import Path
from tesseract_core import Tesseract

import numpy as np

here = Path(__file__).parent.resolve()
print(here)

# from_image = partial(
#     Tesseract.from_image, image_name="autophysics/supersede/inference_jax:0.4.1"
# )
from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")


if __name__ == "__main__":


    CONFIG = here / "inputs/config.yaml"
    SIM_FOLDER = here / "inputs/hvac-cadcore-1/LocalUpload/dfe4b9da6e50d42e0e0c6244d921b655/2025_06_17_HVAC"
    DATASET_FOLDER = here / "inputs/dataset"
    
    with from_api(
        input_path=here / "inputs",
    ) as inference:
        result = inference.apply(
            inputs={
                "config": CONFIG,
                "sim_folder": SIM_FOLDER,
                "dataset_folder": DATASET_FOLDER
            }
        )
        print(result)
