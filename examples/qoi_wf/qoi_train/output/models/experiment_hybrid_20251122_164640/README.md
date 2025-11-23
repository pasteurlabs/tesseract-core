# Experiment: experiment_hybrid_20251122_164640

**Type:** hybrid
**Created:** 2025-11-22T16:46:40.175258

## Directory Structure

```
experiment_hybrid_20251122_164640/
├── config.yaml              # Configuration used for this run
├── experiment_metadata.json # Experiment metadata
├── dataset_info.json        # Dataset split information
├── training_history.json    # Training metrics per epoch
├── model_metrics.json       # Final evaluation metrics
├── models/                  # Trained model files
├── predictions/             # Prediction CSV files
└── plots/                   # Visualization plots
```

## Model Performance

### hybrid_pointnet_small

**train_metrics:**
- mse: 0.015133
- mae: 0.089764
- r2: 0.861160
- rmse: 0.123017
- mape: 48.700060
- max_error: 0.376457
- nmse: 0.138840
- nrmse: 0.372612
- nmae: 0.150743

**val_metrics:**
- mse: 0.009316
- mae: 0.088952
- r2: 0.943451
- rmse: 0.096520
- mape: 147.508981
- max_error: 0.141157
- nmse: 0.056549
- nrmse: 0.237800
- nmae: 0.119361

**test_metrics:**
- mse: 0.014287
- mae: 0.083016
- r2: 0.908371
- rmse: 0.119530
- mape: 73.119687
- max_error: 0.280942
- nmse: 0.091629
- nrmse: 0.302703
- nmae: 0.139596

## Reproducing Results

To reproduce this experiment, use the saved `config.yaml` with the same dataset splits:

```python
# Load the config
config_path = Path('output/models/experiment_hybrid_20251122_164640/config.yaml')

# Load dataset split information
with open('output/models/experiment_hybrid_20251122_164640/dataset_info.json') as f:
    split_info = json.load(f)

# Use the same train/val/test indices to ensure reproducibility
```
