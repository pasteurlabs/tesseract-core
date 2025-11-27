from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from .dataset import cad_collate
from .experiment_tracker import ExperimentTracker
from .models import HybridPointCloudTreeModel
from .utils import set_seed


def train_hybrid_models(
    train_dataset,
    val_dataset,
    test_dataset,
    model_configs: dict[str, dict],
    training_config: dict,
    save_dir: Path,
    config_path: Optional[Path] = None,
    split_info: Optional[dict] = None,
    scaler = None
):
    """Train hybrid PointNeXt + Tree models."""
    # Set random seed for reproducibility
    if split_info and 'seed' in split_info:
        seed = split_info['seed']
        set_seed(seed)
        print(f"🎲 Random seed set to: {seed}")

    if model_configs is None:
        print("No hybrid model configurations found.")
        return

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        base_dir=save_dir,
        experiment_type="hybrid",
        config_path=config_path
    )

    # Log dataset split information
    if split_info:
        tracker.log_dataset_info(split_info)

    # Save scaler to experiment directory for reproducibility
    scaler_path = None
    if scaler is not None:
        scaler_path = tracker.run_dir / "scaler.pkl"
        scaler.save(scaler_path)
        print(f"  ✅ Saved scaler to: {scaler_path.relative_to(save_dir)}")

    # Create PyTorch datasets and loaders (need full data for point clouds)

    batch_size = training_config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=cad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=cad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=cad_collate)

    results = {}

    for name, config in model_configs.items():
        # Reset seed for each model to ensure reproducibility
        if split_info and 'seed' in split_info:
            set_seed(split_info['seed'])
            print(f"🎲 Reset random seed to: {split_info['seed']} for {name}")

        print(f"\n{'='*60}")
        print(f"Training Hybrid Model: {name}")
        print(f"{'='*60}")

        # Create model
        model_config = config.copy()
        model_type = model_config.pop("type")

        if model_type == "hybrid_pc_tree":
            model = HybridPointCloudTreeModel(name=name, **model_config)
            # Attach scaler to model for reproducibility
            model.scaler = scaler
        else:
            print(f"Unknown hybrid model type: {model_type}")
            continue

        # Train model
        model.fit(
            train_data=train_loader,
            val_data=val_loader,
            training_args=training_config
        )

        # Evaluate on all splits
        print("\nEvaluating model...")
        train_metrics = model.evaluate(train_loader)
        val_metrics = model.evaluate(val_loader)
        test_metrics = model.evaluate(test_loader)

        # Log metrics to tracker
        tracker.log_model_metrics(name, train_metrics.__dict__, split="train")
        tracker.log_model_metrics(name, val_metrics.__dict__, split="val")
        tracker.log_model_metrics(name, test_metrics.__dict__, split="test")

        # Print results
        print(f"\n{name} Results:")
        print(f"  Train - MAE: {train_metrics.mae:.6f}, R²: {train_metrics.r2:.6f}")
        print(f"  Val   - MAE: {val_metrics.mae:.6f}, R²: {val_metrics.r2:.6f}")
        print(f"  Test  - MAE: {test_metrics.mae:.6f}, R²: {test_metrics.r2:.6f}")

        # Save model to experiment directory
        model_path = tracker.get_model_path(name, extension=".pkl")
        model.save(model_path)
        print(f"  Model saved to: {model_path}")

        # Store results with metrics and model path
        results[name] = {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "train_metrics": train_metrics.__dict__,
            "val_metrics": val_metrics.__dict__,
            "test_metrics": test_metrics.__dict__,
        }

    # Finalize experiment (saves all metadata and creates README)
    tracker.finalize()

    return results
