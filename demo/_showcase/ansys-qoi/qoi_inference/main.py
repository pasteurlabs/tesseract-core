from pathlib import Path

from tesseract_core import Tesseract

if __name__ == "__main__":
    # Ensure output folder exists
    local_outputs = Path(__file__).parent.resolve() / "outputs"
    local_outputs.mkdir(parents=True, exist_ok=True)

    here = Path("/tesseract/")
    CONFIG = here / "inputs/config.yaml"
    DATASET_FOLDER = here / "inputs/dataset_inference"
    TRAINED_MODEL = here / "inputs/model.pkl"
    SCALER = here / "inputs/scaler.pkl"

    inputs = {
        "config": str(CONFIG),
        "data_folder": str(DATASET_FOLDER),
        "trained_model": str(TRAINED_MODEL),
        "scaler": str(SCALER),
    }

    qoi_inference = Tesseract.from_image(
        "qoi_inference",
        volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"],
    )

    with qoi_inference:
        outputs = qoi_inference.apply(inputs)
