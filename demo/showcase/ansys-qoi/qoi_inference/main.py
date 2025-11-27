from functools import partial
from pathlib import Path

from tesseract_core import Tesseract

here = Path(__file__).parent.resolve()

from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")


if __name__ == "__main__":
    CONFIG = here / "inputs/config.yaml"
    DATASET_FOLDER = here / "inputs/dataset_inference"
    DATASET_FOLDER = DATASET_FOLDER
    DATA = [p.resolve() for p in DATASET_FOLDER.glob("*.npz")]
    TRAINED_MODEL = here / "inputs/model.pkl"
    SCALER = here / "inputs/scaler.pkl"
    # TODO: add scaler as part of the inputs

    print(DATA)

    with from_api(
        input_path=here / "inputs",
    ) as inference:
        result = inference.apply(
            inputs={
                "config": CONFIG,
                "data": DATA,
                "trained_model": TRAINED_MODEL,
                "scaler": SCALER,
            }
        )
        print(result)
