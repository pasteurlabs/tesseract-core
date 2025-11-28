from functools import partial
from pathlib import Path

from tesseract_core import Tesseract

here = Path(__file__).parent.resolve()

from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")


if __name__ == "__main__":
    here = Path("/tesseract/")
    CONFIG = here / "inputs/config.yaml"
    DATASET_FOLDER = here / "inputs/dataset_inference"
    TRAINED_MODEL = here / "inputs/model.pkl"
    SCALER = here / "inputs/scaler.pkl"

    inputs = {
        "config": str(CONFIG),
        "data_folder": str(DATASET_FOLDER),
        "trained_model": str(TRAINED_MODEL),
        "scaler": str(SCALER)
    }

    qoi_train = Tesseract.from_image("qoi_inference", volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"])

    qoi_train.serve()
    outputs = qoi_train.apply(inputs)
    qoi_train.teardown()

    # with from_api(
    #     input_path=here / "inputs",
    # ) as inference:
    #     result = inference.apply(
    #         inputs={
    #             "config": CONFIG,
    #             "data": DATA,
    #             "trained_model": TRAINED_MODEL,
    #             "scaler": SCALER,
    #         }
    #     )
    #     print(result)
