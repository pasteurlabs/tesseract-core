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

    # DATASET TESSERACT
    inputs = {
        "config": str(CONFIG),
        "sim_folder": str(SIM_FOLDER),
        "dataset_folder": str(DATASET_FOLDER),
    }

    qoi_dataset = Tesseract.from_image(
        "qoi_dataset",
        volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"],
    )

    with qoi_dataset:
        outputs = qoi_dataset.apply(inputs)

    # TRAINING TESSERACT
    inputs = {"config": str(CONFIG), "data_folder": str(DATASET_FOLDER)}

    qoi_train = Tesseract.from_image(
        "qoi_train",
        volumes=["./inputs:/tesseract/inputs:ro", "./outputs:/tesseract/outputs:rw"],
    )

    with qoi_train:
        outputs = qoi_train.apply(inputs)

    # INFERENCE TESSERACT

    # Find the latest experiment_hybrid model
    models_dir = local_outputs / "models"
    experiment_dirs = sorted(models_dir.glob("experiment_hybrid_*"))
    latest_experiment = experiment_dirs[-1]
    model_files = list((latest_experiment / "models").glob("*.pkl"))

    # Get relative paths from the project root
    project_root = Path(__file__).parent.resolve()
    TRAINED_MODEL = here / model_files[0].relative_to(project_root)
    SCALER = here / (latest_experiment / "scaler.pkl").relative_to(project_root)

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

    print("\n" + "=" * 80)
    print("✓ Workflow completed successfully!")
    print("=" * 80)
    print("\nOutputs:")
    print(f"  • Dataset:     {DATASET_FOLDER}")
    print(f"  • Models:      {latest_experiment}")
    print(f"  • Predictions: {outputs}")
    print("\n" + "=" * 80)
