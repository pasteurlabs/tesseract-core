import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from .utils import set_seed
from .dataset import cad_collate
from .models import HybridPointCloudTreeModel
from .experiment_tracker import ExperimentTracker


def _worker_init_fn(worker_id):
    """Initialize DataLoader worker with unique seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


@dataclass
class TrainingArgs:
    """Training arguments loaded from config file."""
    # Data
    ds_train: object
    ds_val: object
    split_info: dict
    
    # Model architecture (loaded from config)
    latent_dim: int
    param_fusion: str
    
    # Training hyperparameters (loaded from config)
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    
    # Hardware (loaded from config)
    cpu: bool
    no_amp: bool
    workers: int
    
    # Paths
    models_dir: str
    
    # Random seed
    seed: int


def create_training_args_from_config(config: dict, train_dataset, val_dataset, split_info, models_dir: str) -> TrainingArgs:
    """Create TrainingArgs from config dictionary."""
    model_config = config.get("model_spec", {})
    training_config = config.get("training", {})
    
    return TrainingArgs(
        # Data
        ds_train=train_dataset,
        ds_val=val_dataset,
        split_info=split_info,
        
        # Model architecture from config
        latent_dim=model_config.get("latent_dim", 8),
        param_fusion=model_config.get("param_fusion", "concat"),
        
        # Training hyperparameters from config
        batch_size=training_config.get("batch_size", 32),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 1e-4),
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 20),
        
        # Hardware from config
        cpu=training_config.get("cpu", False),
        no_amp=training_config.get("no_amp", False),
        workers=training_config.get("workers", 4),
        
        # Paths
        models_dir=str(models_dir),
        
        # Random seed
        seed=config.get("random_seed", 42),
    )


def train_hybrid_models(
    train_dataset,
    val_dataset,
    test_dataset,
    model_configs: dict[str, dict],
    training_config: dict,
    save_dir: Path,
    config_path: Optional[Path] = None,
    split_info: Optional[dict] = None
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

    # Create PyTorch datasets and loaders (need full data for point clouds)

    batch_size = training_config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=cad_collate, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=cad_collate, worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=cad_collate, worker_init_fn=_worker_init_fn)

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
        # elif model_type == "hybrid_pc_nn":
        #     model = HybridPointCloudNNModel(name=name, **model_config)
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

        # Store results
        results[name] = {
            "train_metrics": train_metrics.__dict__,
            "val_metrics": val_metrics.__dict__,
            "test_metrics": test_metrics.__dict__,
        }

        # Print results
        print(f"\n{name} Results:")
        print(f"  Train - MAE: {train_metrics.mae:.6f}, R²: {train_metrics.r2:.6f}")
        print(f"  Val   - MAE: {val_metrics.mae:.6f}, R²: {val_metrics.r2:.6f}")
        print(f"  Test  - MAE: {test_metrics.mae:.6f}, R²: {test_metrics.r2:.6f}")

        # Save model to experiment directory
        model_path = tracker.get_model_path(name, extension=".pkl")
        model.save(model_path)
        print(f"  Model saved to: {model_path}")

    # Finalize experiment (saves all metadata and creates README)
    tracker.finalize()

    #return results
    return model_path