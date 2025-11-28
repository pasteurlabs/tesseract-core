from functools import partial
from pathlib import Path

from tesseract_core import Tesseract

here = Path(__file__).parent.resolve()

#from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")
    
here = Path("/tesseract/")
if __name__ == "__main__":
    CONFIG = here / "inputs/config.yaml"
    SIM_FOLDER = here / "inputs/Ansys_Runs"
    OUTPUT_DIR = here / "outputs"
    #OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_FOLDER = OUTPUT_DIR / "dataset"

    # FROM API
    # from_api = partial(Tesseract.from_tesseract_api, tesseract_api="tesseract_api.py")
    # with from_api(
    #     input_path=here / "inputs",
    # ) as inference:
    #     result = inference.apply(
    #         inputs={
    #             "config": CONFIG,
    #             "sim_folder": SIM_FOLDER,
    #             "dataset_folder": DATASET_FOLDER,
    #         }
    #     )
    #     print(result)

    #  FROM IMAGE
    inputs={
        "config": str(CONFIG),
        "sim_folder": str(SIM_FOLDER),
        "dataset_folder": str(DATASET_FOLDER),
    }

    qoi_dataset = Tesseract.from_image("qoi_dataset", volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"])

    qoi_dataset.serve()
    outputs = qoi_dataset.apply(inputs)
    qoi_dataset.teardown()



