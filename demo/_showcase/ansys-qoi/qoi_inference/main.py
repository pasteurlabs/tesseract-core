from pathlib import Path

from tesseract_core import Tesseract

if __name__ == "__main__":
    # Ensure output folder exists
    here = Path(__file__).parent.resolve()
    OUTPUT_DIR = here / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CONFIG = "config.yaml"
    DATASET_FOLDER = "dataset_inference"
    TRAINED_MODEL = "model.pkl"
    SCALER = "scaler.pkl"

    inputs = {
        "config": CONFIG,
        "data_folder": DATASET_FOLDER,
        "trained_model": TRAINED_MODEL,
        "scaler": SCALER,
    }

    qoi_inference = Tesseract.from_image(
        "qoi_inference", input_path="./inputs", output_path="./outputs"
    )

    with qoi_inference:
        outputs = qoi_inference.apply(inputs)
