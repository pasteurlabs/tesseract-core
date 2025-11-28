# QoI-based Workflows with Ansys Fluent and Terreract

This showcase demonstrates an end-to-end workflow for building QoI-based surrogate models from Ansys simulation data using Tesseract. The overall workflow is illustrated below and demonstrated within our [QoI-based surrogacy showcase](https://si-tesseract.discourse.group/t/qoi-based-workflows-with-ansys-fluent-and-terreract/110).

## Overview

This workflow implements a **QoI-based surrogate modeling pipeline** that:
1. Processes Ansys simulation runs (CAD files + simulation results)
2. Extracts point clouds, CAD parameters, boundary conditions, and Quantities of Interest (QoI)
3. Trains hybrid ML models (PointNet + Random Forest) to predict QoI
4. Performs inference on new CAD designs

### Workflow Components

The workflow consists of three Tesseract components:

```
+------------------+
|   qoi_dataset    |  <- Processes Ansys simulation runs
|   (Data Gen)     |  -> Generates NPZ datasets
+--------+---------+
         |
         v
+------------------+
|    qoi_train     |  <- Trains ML models on processed data
|   (Training)     |  -> Outputs trained models + scalers
+--------+---------+
         |
         v
+------------------+
|  qoi_inference   |  <- Import new geometry design
|   (Inference)    |  -> Generates QoI predictions
+------------------+
```
## Input Data

The workflow expects **Ansys simulation runs** in the following structure:

```
inputs/Ansys_Runs/
├── Run_0001/
│   ├── geometry.stl              # CAD geometry file
│   ├── metadata.json.series      # BC and CAD parameteres
│   ├── all_pressure.txt          # QoI outputs
│   └── ...
├── Run_0002/
│   └── ...
└── ...
```

Each simulation run directory should contain:
- **CAD file** (`.stl` format): 3D geometry for point cloud sampling
- **Boundary condition data**: Parameters varied across simulations
- **CAD parameters**: Parameters used during the CAD design process
- **Simulation results**: Target QoI values to predict

## Configuration

Each component uses a `config.yaml` file in `inputs/config.yaml` that specifies:
- Point cloud sampling strategy
- Parameter extraction rules
- QoI definitions
- Model architecture and training hyperparameters

## Running the Workflow

### Option 1: Run Complete Workflow

Execute all three components sequentially:

```bash
cd demo/_showcase/ansys-qoi
tesseract build ./qoi_dataset/
tesseract build ./qoi_inference/
tesseract build ./qoi_train/
python workflow.py
```

This will:
1. Process all Ansys runs into NPZ datasets
2. Train QoI-based surrogate models on the dataset
3. Run inference using the latest trained model

**Output locations:**
- Dataset: `outputs/dataset/*.npz`
- Models: `outputs/models/experiment_hybrid_YYYYMMDD_HHMMSS/`
- Predictions: `outputs/predictions_YYYYMMDD_HHMMSS.csv`

### Option 2: Run Components Individually

#### 1. Dataset Generation (`qoi_dataset`)

Process Ansys simulation runs into point cloud datasets:

```bash
cd qoi_dataset
tesseract build .
python main.py
```

**What it does:**
- Samples point clouds from CAD files (`.stl`)
- Extracts boundary condition parameters
- Extracts CAD sketch design parameters
- Parses QoI from simulation outputs
- Generates compressed NPZ files with all data

**Outputs:** `outputs/dataset/*.npz` files

#### 2. Model Training (`qoi_train`)

Train hybrid ML models on the processed dataset:

```bash
cd qoi_train
tesseract build .
python main.py
```

**What it does:**
- Loads NPZ dataset files
- Creates train/val/test splits
- Fits data scaler
- Trains QoI-based surrogate model
- Evaluates model performance
- Saves trained models and scalers

**Outputs:**
- `outputs/models/experiment_hybrid_YYYYMMDD_HHMMSS/`
  - `models/hybrid_pointnet_small.pkl` - Trained model weights
  - `scaler.pkl` - Data normalization scaler
  - `config.yaml` - Training configuration
  - `model_metrics.json` - Performance metrics

#### 3. Inference (`qoi_inference`)

Run predictions on new geometries using trained models:

```bash
cd qoi_inference
tesseract build .
python main.py
```

**What it does:**
- Loads trained model and scaler
- Processes input geometries
- Generates QoI predictions
- Saves predictions to CSV

**Outputs:**
- `outputs/predictions_YYYYMMDD_HHMMSS.csv`
