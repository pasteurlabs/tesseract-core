from pathlib import Path

from tesseract_core import Tesseract

if __name__ == "__main__":
    # Ensure output folder exists
    local_outputs = Path(__file__).parent.resolve() / "outputs"
    local_outputs.mkdir(parents=True, exist_ok=True)

    here = Path("/tesseract/")
    CONFIG = here / "inputs/config.yaml"
    SIM_FOLDER = here / "inputs/Ansys_Runs"
    OUTPUT_DIR = here / "outputs"
    DATASET_FOLDER = OUTPUT_DIR / "dataset"

    inputs = {
        "config": str(CONFIG),
        "sim_folder": str(SIM_FOLDER),
        "dataset_folder": str(DATASET_FOLDER),
    }

    qoi_dataset = Tesseract.from_image(
        "qoi_dataset",
        volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"],
    )

    qoi_dataset.serve()
    outputs = qoi_dataset.apply(inputs)
    qoi_dataset.teardown()
