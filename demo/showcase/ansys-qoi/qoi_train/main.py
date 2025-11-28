from functools import partial
from pathlib import Path

from tesseract_core import Tesseract


from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")


if __name__ == "__main__":
    here = Path(__file__).parent.resolve()
    CONFIG = here / "inputs/config.yaml"
    DATASET_FOLDER = here / "inputs/dataset_reduced"
    print(str(DATASET_FOLDER))
    # Convert paths to strings (Tesseract will handle them as external references)
    
    #DATA = [str(p.resolve()) for p in DATASET_FOLDER.glob("*.npz")]
    #print(f"Found {len(DATA)} dataset files")

    # with from_api(
    #     input_path=here / "inputs",
    # ) as inference:
    #     result = inference.apply(inputs={"config": str(CONFIG), "data_folder": str(DATASET_FOLDER)})
    #     print(result)

    # here = Path("/tesseract/")
    
    

    here = Path("/tesseract/")
    CONFIG = here / "inputs/config.yaml"
    DATASET_FOLDER = here / "inputs/dataset_reduced"
    inputs = {
        "config": str(CONFIG),
        "data_folder": str(DATASET_FOLDER)
    }

    qoi_train = Tesseract.from_image("qoi_train", volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"])

    qoi_train.serve()
    outputs = qoi_train.apply(inputs)
    qoi_train.teardown()