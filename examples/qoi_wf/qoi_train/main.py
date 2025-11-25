from functools import partial
from pathlib import Path

from tesseract_core import Tesseract

here = Path(__file__).parent.resolve()

from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")


if __name__ == "__main__":
    CONFIG = here / "inputs/config.yaml"
    DATASET_FOLDER = here / "../qoi_dataset/outputs/dataset"

    # Convert paths to strings (Tesseract will handle them as external references)
    DATA = [str(p.resolve()) for p in DATASET_FOLDER.glob("*.npz")]
    print(f"Found {len(DATA)} dataset files")

    with from_api(
        input_path=here / "inputs",
    ) as inference:
        result = inference.apply(inputs={"config": CONFIG, "data": DATA})
        print(result)
