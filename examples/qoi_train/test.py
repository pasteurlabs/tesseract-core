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
    DATASET_FOLDER = here / "../qoi_dataset/inputs/dataset"

    
    with from_api(
        input_path=here / "inputs",
    ) as inference:
        result = inference.apply(
            inputs={
                "config": CONFIG,
                "dataset_folder": DATASET_FOLDER
            }
        )
        print(result)
