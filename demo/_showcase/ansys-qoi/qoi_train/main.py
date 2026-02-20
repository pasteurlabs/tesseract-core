from pathlib import Path

from tesseract_core import Tesseract

if __name__ == "__main__":
    # Ensure output folder exists
    here = Path(__file__).parent.resolve()
    OUTPUT_DIR = here / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CONFIG = "config.yaml"
    DATASET_FOLDER = "dataset_reduced"
    inputs = {"config": CONFIG, "data_folder": DATASET_FOLDER}

    qoi_train = Tesseract.from_image(
        "qoi_train", input_path="./inputs", output_path="./outputs"
    )

    with qoi_train:
        outputs = qoi_train.apply(inputs)
