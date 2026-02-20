from pathlib import Path

from tesseract_core import Tesseract

if __name__ == "__main__":
    # Ensure output folder exists
    here = Path(__file__).parent.resolve()
    OUTPUT_DIR = here / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CONFIG = "config.yaml"
    SIM_FOLDER = "Ansys_Runs"
    DATASET_FOLDER = "dataset"

    inputs = {
        "config": CONFIG,
        "sim_folder": SIM_FOLDER,
        "dataset_folder": DATASET_FOLDER,
    }

    qoi_dataset = Tesseract.from_image(
        "qoi_dataset",
        input_path="./inputs",
        output_path="./outputs",
    )

    with qoi_dataset:
        outputs = qoi_dataset.apply(inputs)
