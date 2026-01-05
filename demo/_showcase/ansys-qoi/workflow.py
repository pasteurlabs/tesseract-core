import shutil
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

    # DATASET TESSERACT
    inputs = {
        "config": CONFIG,
        "sim_folder": SIM_FOLDER,
        "dataset_folder": DATASET_FOLDER,
    }

    qoi_dataset = Tesseract.from_image(
        "qoi_dataset", input_path="./inputs", output_path="./outputs"
    )

    with qoi_dataset:
        outputs = qoi_dataset.apply(inputs)

    # TRAINING TESSERACT
    # Copy outputs to input folder
    shutil.copytree(
        here / "outputs" / DATASET_FOLDER,
        here / "inputs" / DATASET_FOLDER,
        dirs_exist_ok=True,
    )

    inputs = {"config": CONFIG, "data_folder": DATASET_FOLDER}

    qoi_train = Tesseract.from_image(
        "qoi_train", input_path="./inputs", output_path="./outputs"
    )
    with qoi_train:
        outputs = qoi_train.apply(inputs)

    # INFERENCE TESSERACT

    # Find the latest experiment_hybrid model
    models_dir = OUTPUT_DIR / "models"
    experiment_dirs = sorted(models_dir.glob("experiment_hybrid_*"))
    latest_experiment = experiment_dirs[-1]
    model_file = next(iter((latest_experiment / "models").glob("*.pkl")))

    TRAINED_MODEL = model_file.name
    SCALER = "scaler.pkl"

    # Copy outputs to input folder
    shutil.copy(model_file, here / "inputs" / TRAINED_MODEL)
    shutil.copy(latest_experiment / SCALER, here / "inputs" / SCALER)

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

    print("\n" + "=" * 80)
    print("✓ Workflow completed successfully!")
    print("=" * 80)
    print("\nOutputs:")
    print(f"  • Dataset:     {DATASET_FOLDER}")
    print(f"  • Models:      {latest_experiment}")
    print(f"  • Predictions: {outputs}")
    print("\n" + "=" * 80)
