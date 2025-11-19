from functools import partial
from pathlib import Path

from tesseract_core import Tesseract

import numpy as np

here = Path(__file__).parent.resolve()

# from_image = partial(
#     Tesseract.from_image, image_name="autophysics/supersede/inference_jax:0.4.1"
# )
from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")

if __name__ == "__main__":

    xyz = np.ones((1000, 3))
    normals = np.ones((1000, 3))
    params = np.ones(1000)
    
    with from_api(
        input_path=here / "inputs",
    ) as inference:
        result = inference.apply(
            inputs={
                "xyz": xyz,
                "normals" : normals,
                "params" : params,
                "trained_model": "dummy.txt",
            }
        )
        print(result)
