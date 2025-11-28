from pathlib import Path

from tesseract_core import Tesseract

if __name__ == "__main__":
    # Ensure output folder exists
    local_outputs = Path(__file__).parent.resolve() / "outputs"
    local_outputs.mkdir(parents=True, exist_ok=True)

    here = Path("/tesseract/")
    CONFIG = here / "inputs/config.yaml"
    DATASET_FOLDER = here / "inputs/dataset_reduced"
    inputs = {"config": str(CONFIG), "data_folder": str(DATASET_FOLDER)}

    qoi_train = Tesseract.from_image(
        "qoi_train",
        volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"],
    )

    qoi_train.serve()
    outputs = qoi_train.apply(inputs)
    qoi_train.teardown()
