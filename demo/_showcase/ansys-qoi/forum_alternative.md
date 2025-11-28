# Ansys <-> Tesseract: Exploring QoI Workflows

## Table of Contents

[1. Case Study: HVAC Duct Dataset](#1-case-study-hvac-duct-dataset)
   - [1.1. Dataset Variations](#11-dataset-variations)
   - [1.2. QoI](#12-qoi)
   - [1.3. Dataset Summary](#13-dataset-summary)

[2. QoI Workflows](#2-qoi-workflows)
   - [2.1. CAD geometries + boundary conditions → QoI-based surrogacy](#21-cad-geometries--boundary-conditions--qoi-based-surrogacy)
   - [2.2. CAD geometries + boundary conditions → Full-field surrogacy → QoI](#22-cad-geometries--boundary-conditions--full-field-surrogacy--qoi)

[3. Ansys↔Tesseract QoI Workflow Proposal](#3-ansystesseract-qoi-workflow-proposal)
   - [3.1. Tesseract Components and Workflows](#31-tesseract-workflows-and-components)
     - [3.1.1. Tesseract Components](#311-tesseract-components)
     - [3.1.2. Tesseract Training Workflow](#312-tesseract-training-workflow)
     - [3.1.3. Tesseract Inference Workflow](#313-tesseract-inference-workflow)

[4. Results](#4-results)

[5. Outlook](#5-outlook)

[Appendix](#appendix)
- [A. QoI-based Surrogate Model](#a-qoi-based-surrogate-model)
  - [A.1. Dataset Preparation](#a1-dataset-preparation)
  - [A.2. Model Architecture](#a2-model-architecture)

Design iteration is expensive. Engineers adjust CAD parameters and boundary conditions, run simulations, extract key metrics, and repeat until requirements are met. These critical metrics—static pressures, flow rates, stress concentrations—are what we call Quantities of Interest (QoI). They directly determine whether a design passes or fails.

The bottleneck? Each iteration requires meshing, solving, and post-processing. For complex geometries, this means hours or days per design variant.

This showcase demonstrates a practical alternative: training surrogate models that predict QoI directly from CAD geometry and boundary conditions, bypassing expensive simulations during design exploration. We use an HVAC duct dataset generated with Ansys Fluent and show how Tesseract's component framework simplifies building and deploying these workflows.

**What you'll see:**
1. How we assembled an internal aerodynamics dataset from ~300 Ansys Fluent simulations
2. How a QoI-based surrogate model achieves R² > 0.95 on pressure predictions
3. How Tesseract components make these workflows modular, reproducible, and deployment-ready

## 1. Case Study: HVAC Duct Dataset

We generated approximately 300 Ansys Fluent simulations of internal airflow through an HVAC duct with two baffle plates. The geometry is representative of real-world HVAC systems where flow distribution and pressure management are critical design constraints.

![alt text](images/duct_1.png)

**Key features:**
- Inlet baffle plate constrains and conditions the incoming flow
- Middle baffle plate redirects flow at the branch intersection

### 1.1 Dataset Variations

**CAD parameters**

Four geometric parameters control the duct configuration:
- `d13`: Angle between duct branches
- `d34`: Curvature angle of the duct branch
- `d61`: Angle of the middle baffle plate
- `d72`: Aperture length of the inlet baffle plate

<table> <tr> <td align="center"> <img src="images/d13.png" alt="d13" width="400"/><br/> <b>d13:</b> Angle between duct branches </td> <td align="center"> <img src="images/d34.png" alt="d34" width="400"/><br/> <b>d34:</b> Curvature angle of duct branch </td> </tr> <tr> <td align="center"> <img src="images/d61.png" alt="d61" width="400"/><br/> <b>d61:</b> Angle of middle baffle plate </td> <td align="center"> <img src="images/d72.png" alt="d72" width="400"/><br/> <b>d72:</b> Aperture length of inlet baffle </td> </tr> </table>

**Boundary conditions**

Inlet velocity magnitude was varied to capture different flow regimes and operating conditions.

### 1.2. QoI

Each simulation output includes static pressure values averaged at four measurement planes: `inlet`, `outlet`, `p2-plane` (downstream of inlet baffle), and `p3-plane` (upstream of middle baffle). These pressure values serve as our QoI—the metrics an engineer would use to evaluate design performance.

While additional metrics like pressure drop (Δp) between planes can be derived from these base values, we focus on the four pressure measurements for this showcase.

<table> <tr> <td align="center"> <img src="images/p2-plane.png" alt="p2-plane" height="300"/><br/> <b>p2-plane:</b> YZ plane after inlet baffle plate </td> <td align="center"> <img src="images/p3-plane.png" alt="p3-plane" height="300"/><br/> <b>p3-plane:</b> YZ plane before middle baffle plate </td> </tr> </table>

<!-- > 📝 **Note:** How Ansys Fluent reporting tool enables the extraction of QoI
`/report/surface-integrals area-weighted-avg inlet outlet p2-plane p3-plane () pressure yes all_pressure.txt`-->

### 1.3. Dataset Summary
<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <th style="border: none; border-bottom: 2px solid #ddd; padding: 8px;"><b>CAD Parameters</b></th>
    <th style="border: none; border-bottom: 2px solid #ddd; padding: 8px;"><b>BC Parameters</b></th>
    <th style="border: none; border-bottom: 2px solid #ddd; padding: 8px;"><b>Base QoI</b></th>
  </tr>
  <tr style="border: none;">
    <td style="border: none; padding: 8px;"><code>d61</code>: Angle of middle baffle plate</td>
    <td style="border: none; padding: 8px;">Inlet velocity</td>
    <td style="border: none; padding: 8px;">Static pressure inlet</td>
  </tr>
  <tr style="border: none;">
    <td style="border: none; padding: 8px;"><code>d72</code>: Aperture length of inlet baffle plate</td>
    <td style="border: none; padding: 8px;"></td>
    <td style="border: none; padding: 8px;">Static pressure outlet</td>
  </tr>
  <tr style="border: none;">
    <td style="border: none; padding: 8px;"><code>d13</code>: Angle between ducts</td>
    <td style="border: none; padding: 8px;"></td>
    <td style="border: none; padding: 8px;">Static pressure p2-plane</td>
  </tr>
  <tr style="border: none;">
    <td style="border: none; padding: 8px;"><code>d34</code>: Curvature angle of duct branch</td>
    <td style="border: none; padding: 8px;"></td>
    <td style="border: none; padding: 8px;">Static pressure p3-plane</td>
  </tr>
</table>


## 2. QoI Workflows

Understanding how design parameters affect QoI is essential for efficient engineering. A workflow that maps CAD files directly to QoI eliminates meshing, solving, and post-processing steps—drastically reducing iteration time during early-stage design.

Before showing how Tesseract enables this workflow with Ansys data, let's examine two architectural approaches.

### 2.1. CAD geometries + boundary conditions → QoI-based surrogacy
<p align="center">
  <img src="images/qoi_overview_best.png">
</p>

This direct approach trains a model to predict QoI from geometry and boundary conditions. We selected this workflow for the showcase because it's practical for early-stage design: straightforward to implement, computationally lightweight, and easier to maintain than full-field approaches. The model learns patterns in how geometric variations affect key performance metrics without reconstructing entire flow fields.

### 2.2. CAD geometries + boundary conditions → Full-field surrogacy → QoI
<p align="center">
  <img src="images/qoi_overview_full.png">
</p>

Alternatively, a full-field surrogate model can predict entire pressure distributions, temperature fields, or stress contours. QoI are then extracted from these fields using standard post-processing. While this provides complete field data for detailed analysis, it introduces significant complexity—larger models, more training data requirements, and higher computational overhead. For early-stage design where quick iteration matters more than full field visibility, the direct QoI approach is often more appropriate.

## 3. Ansys<->Tesseract QoI Workflow Proposal

Tesseract provides a framework for packaging simulation and machine learning workflows into modular, reusable components. Each component declares explicit inputs and outputs, handles its own dependencies, and can be composed into larger workflows. For QoI prediction on Ansys data, this means separating data preprocessing, model training, and inference into independent units that can be developed, tested, and deployed separately.

The diagram below shows how three Tesseract components orchestrate training and inference for QoI-based surrogacy on HVAC simulation data.

![alt text](images/tesseract_wf.png)

We define two workflows: a training workflow where the surrogate model learns from historical Ansys Fluent data, and an inference workflow where the trained model predicts QoI for new designs.

### 3.1. Tesseract Workflows and Components
#### 3.1.1 Tesseract Components

Tesseract Components address common pain points in simulation-ML pipelines. Dependencies are encapsulated within each component—no need to manage conflicting versions of geometric processing libraries or deep learning frameworks across your team. This matters when working with rapidly evolving tools in geometric deep learning and CAD processing.

Components are also deployment-ready by design. The same component used during research runs identically in production. Version control, input validation, and error handling are built in.

Below are the three components used in this workflow.

**Dataset Tesseract**

Ingests raw Ansys Fluent simulation outputs and produces machine-learning-ready datasets.

### TODO UPDATE LAST AFTER ALESSANDRO REVIEW
```python
class InputSchema(BaseModel):
    config: InputFileReference = Field(description="Configuration file")
    sim_folder: str | Path = Field(
        description="Folder path containing Ansys Fluent simulations with CAD files and QoI reports",
    )
    dataset_folder: str | Path = Field(
        description="Folder path where pre-processed simulations will be dumped into"
    )
```
```python
class OutputSchema(BaseModel):
    data: list[OutputFileReference] = Field(
        description="List of npz files containing List of npz files containing point-cloud data information,  simulation parameters and QoIs",
    )
```

**Training Tesseract**

Consumes preprocessed data and trains the QoI prediction model.

```python
class InputSchema(BaseModel):
    config: InputFileReference = Field(description="Configuration file")
    data: list[str] = Field(
        description="List of npz files containing List of npz files containing point-cloud data information,  simulation parameters and QoIs"
    )
```
```python
class OutputSchema(BaseModel):
    trained_models: list[OutputFileReference] = Field(
        description="Pickle file containing weights of trained model"
    )
    scalers: list[OutputFileReference] = Field(
        description="Pickle file containing the scaling method for the dataset"
    )
```

**Inference Tesseract**

Applies the trained model to new geometries and boundary conditions.

```python
class InputSchema(BaseModel):
    config: InputFileReference = Field(description="Configuration file")
    data: list[str | Path] = Field(
        description="List of npz files containing point-cloud data information and simulation parameters"
    )
    trained_model: InputFileReference = Field(
        description="Pickle file containing weights of trained model"
    )
    scaler: InputFileReference = Field(
        description="Pickle file containing the scaling method for the dataset"
    )
```
```python
class OutputSchema(BaseModel):
    qoi: Array[(None, None), Float32] = Field(
        description="QoIs - 2D array where each row is a prediction",
    )
```

#### 3.1.2. Tesseract Training Workflow

Two components form the training pipeline:

- **Dataset Pre-Processing Tesseract**: Extracts point cloud samples from STL files, reads CAD parameters and boundary conditions from metadata, and parses QoI from Ansys report files. Outputs standardized NPZ files.

- **Training Tesseract**: Loads the preprocessed datasets and trains the surrogate model to learn the functional relationship between geometry, boundary conditions, and QoI.

#### 3.1.3. Tesseract Inference Workflow

The inference workflow enables QoI prediction without running simulations:

- **Dataset Pre-Processing Tesseract**: Applies the same preprocessing to new CAD files and boundary conditions, ensuring format compatibility with the trained model.

- **Inference Tesseract**: Feeds preprocessed data through the trained model to predict QoI.

## 4. Results

Before examining model performance, it's worth understanding what features the model actually sees. The preprocessing pipeline transforms raw Ansys outputs into these inputs:

- **Point cloud coordinates and normals**: (x, y, z) positions and (nx, ny, nz) normal vectors sampled from the STL geometry
- **Point cloud-derived parameters**: Geometric descriptors computed from the sampled points (bounding box dimensions, centroid, etc.)
- **CAD parameters**: The sketch parameters that define the geometry (`d61`, `d72`, `d13`, `d34`)
- **Boundary conditions**: Inlet velocity
- **QoI** (training only): Averaged static pressures at the four measurement planes

More details on preprocessing are in [Appendix A.1.](#a1-dataset-preparation)

![alt text](images/full_model.png)

The model architecture is straightforward: a PointNet-based shape embedder processes point coordinates and normals into a compact geometric representation. This embedding is concatenated with boundary conditions, point cloud-derived parameters, and CAD parameters, then fed to a Random Forest regressor for QoI prediction. See [Appendix A.2.](#a2-model-architecture) for architecture details.

**Performance on held-out test set (31 samples):**
```
"test_metrics": {
  "r2": 0.9559590920924331,
  "nmse": 0.04404090592149448,
  "nrmse": 0.20985925815702633,
  "nmae": 0.07607935409954629
}
```

These metrics indicate reliable QoI predictions, particularly for early-stage design where approximate values enable rapid exploration of the design space.

An alternative model trained without CAD parameters is described in [Appendix A.2.](#a2-model-architecture) This scenario represents cases where original CAD sketches have been lost—a common occurrence when design history is poorly maintained or files are transferred between CAD systems. Even without explicit parameter values, the shape embedder extracts sufficient geometric information from the STL to achieve reasonable accuracy (R² = 0.90).

## 5. Outlook

This showcase demonstrates a practical workflow for QoI prediction from CAD geometry using Ansys simulation data and Tesseract components. The approach offers tangible benefits for simulation-driven design:

**Accelerated Design Iteration**: Direct QoI prediction bypasses expensive simulation runs during early-stage exploration. Engineers can rapidly evaluate hundreds of design variants, narrowing the design space before committing to high-fidelity analysis. For the HVAC case shown here, QoI prediction is near-instantaneous compared to Fluent solve times.

**Design Automation and Optimization**: With fast QoI prediction, engineers can integrate surrogate models into automated design loops. Optimization algorithms can explore parameter spaces that would be prohibitively expensive with full simulations. This enables systematic design space exploration rather than intuition-driven iteration.

**Differentiable Design Workflows**: Tesseract's support for differentiable programming opens possibilities beyond forward prediction. Gradient-based optimization can find optimal CAD parameters for target QoI values. Sensitivity analysis reveals which geometric features most strongly influence performance. Design tradeoffs become quantifiable—for example, understanding how baffle angle affects pressure drop versus flow uniformity.

**Handling Missing Design Information**: The alternative model architecture (Appendix A.2) demonstrates that QoI prediction remains viable even when CAD parameter history is unavailable. This addresses a practical reality in engineering organizations: STL files are often exchanged without underlying parametric models. The shape embedder extracts geometric features directly from mesh topology, enabling surrogate modeling even with incomplete design provenance.

**Transferability to Other Physics**: While this showcase focuses on CFD pressure predictions, the workflow architecture applies broadly. Similar approaches can be built for structural analysis (stress/displacement QoI), thermal management (temperature QoI), or multiphysics problems. The Tesseract component structure remains the same—only the preprocessing specifics and model architectures change to suit the physics domain.

**Hybrid Simulation-ML Strategies**: Surrogate models don't replace high-fidelity simulation—they augment it. Engineers can use ML models for rapid screening, then validate promising designs with full Ansys simulations. This hybrid approach focuses expensive compute resources where they matter most: final design verification rather than exploratory iterations.

**Integration with Existing Workflows**: The Tesseract components integrate with standard Ansys outputs (STL, mesh files, report files) without requiring changes to existing simulation templates or practices. Teams can adopt QoI-based workflows incrementally, training models on historical simulation data already in their archives.

The key enabler is modularity. By packaging data preprocessing, training, and inference into separate Tesseract components, organizations can build libraries of reusable workflow elements. A new QoI-based project can leverage existing geometry preprocessing while swapping in problem-specific physics models. Version control and reproducibility come standard.

## Appendix
### A. QoI-based Surrogate Model
#### A.1. Dataset Preparation

Each Ansys simulation run produces these files:
```
Experiment_0
|-duct_baffle.stl
|-duct_baffle.msh.h5
|-duct_baffle.cas.h5
|-duct_baffle.dat.h5
|-metadata.json.series
|-all_pressure.txt
```

Feature extraction from these files is critical for training. Here's how each data type is processed:

##### CAD Geometry File Pre-Processing

Point cloud representation offers a good balance between geometric fidelity and implementation simplicity. Other approaches (voxelization, signed distance fields, graph representations) are possible but add complexity.

*Point-Cloud Sampling (Points & Normals)*

The sampling strategy retrieves both point coordinates and normal vectors from the STL surface. Similar to Ansys Meshing's "Sphere of Influence" feature, the implementation can increase sampling density in regions of interest. Since the baffle plates are geometrically important, spheres of influence around these areas concentrate more points where they matter.

<table>
  <tr>
    <td align="center">
      <img src="images/poisson_sampling.png" alt="Image 1" height="150"/><br/>
      <b></b> Poisson sampling (2048 pts)
    </td>
    <td align="center">
      <img src="images/poisson_soi.png" alt="Image 2" height="150"/><br/>
      <b></b> Poisson sampling with SoI (2048 pts)
    </td>
    <td align="center">
      <img src="images/poisson_plus_soi.png" alt="Image 3" height="150"/><br/>
      <b></b> Poisson sampling with SoI (4096 pts)
    </td>
  </tr>
</table>

Sampling parameters are controlled via `config.yaml`.

*Point-Cloud Derived Parameters*

Several geometric descriptors can be computed directly from sampled points, providing global shape information:

- min(x, y, z): minimum coordinate values
- max(x, y, z): maximum coordinate values
- size: bounding box dimensions [max(xyz) – min(xyz)]
- diag: bounding box diagonal length
- max_side: maximum bounding box extent
- centroid: geometric center of point cloud

These features help the model capture coarse geometric attributes that influence QoI.

##### CAD Parameters Pre-Processing

The `metadata.json.series` file records parametric variations applied during dataset generation:

```json
"variations": {
    "d61": 0.8,
    "d13": 1.7,
    "d72": 0.6,
    "d34": 2.7,
    ...
}
```

These values are extracted directly from the metadata file.

##### Boundary Conditions Pre-Processing

Boundary condition variations are also stored in `metadata.json.series`:

```json
"variations": {
    ...
    "mean_velocity": 78.0
}
```

##### QoI Pre-Processing

Ansys Fluent simulation reports contain static pressure values at the specified measurement planes (see [Section 1.2](#12-qoi)):

```
                         "Surface Integral Report"
           Area-Weighted Average
                 Static Pressure                 [Pa]
-------------------------------- --------------------
                           inlet             20239503
                          outlet           -82535.753
                        p2-plane           -209300.83
                        p3-plane           -318261.65
                ---------------- --------------------

```

A dedicated `SurfaceIntegralReport` class parses these files to extract pressure data.

##### Storing Pre-Processed Dataset

After preprocessing, all data (point cloud coordinates and normals, geometric parameters, CAD parameters, boundary conditions, and QoI) are consolidated and serialized to NPZ format for efficient storage. These NPZ files are then converted to PyTorch Dataset objects for training and inference.

#### A.2. Model Architecture

The architecture employs a geometry embedding strategy: point cloud coordinates and normals are processed through a neural network to create a compact latent representation `z` that captures essential geometric features. This embedding is concatenated with a parameter vector `p` containing physics and design information:

**Parameter vector `p` components:**
- Core (always included): Boundary conditions (inlet velocity)
- Optional (model-variant dependent):
   - Point-cloud derived parameters: `["min", "max", "size", "diag", "max_side", "centroid"]`
   - CAD parameters: `["d61", "d72", "d13", "d34"]` (full set or subset)

After concatenation of `p` and `z`, a predictor outputs the QoI vector `q`. This showcase uses a Random Forest regressor for its simplicity and robust performance on small datasets.

![alt text](images/full_model.png)

**Shape Embedder Architecture**

The model employs a PointNet-based embedder for 3D point cloud processing, chosen for its effectiveness with limited data due to relatively few parameters compared to more sophisticated alternatives. The architecture consists of three stages:

1. **Point-wise Feature Extraction**: A Multi-Layer Perceptron (MLP) implemented as 1D convolutions processes each point independently, extracting local geometric features from coordinates and normals (6D input).

2. **Global Feature Aggregation**: Max pooling aggregates point-wise features into a single global descriptor, capturing the most prominent features across the entire point cloud while maintaining permutation invariance.

3. **Latent Space Projection**: A fully connected MLP projects the global features into a lower-dimensional latent representation (typically 8-16D).

The network incorporates batch normalization for training stability, dropout (0.2) for regularization, and Xavier initialization for optimal weight initialization.

More sophisticated point cloud embedders such as PointNeXt and PointBERT were evaluated during development. However, these architectures showed reduced performance, likely because their increased complexity and parameter count are poorly suited to datasets of this size (~300 samples).

**Baseline model**

The baseline model includes boundary conditions, point-cloud derived parameters, and CAD parameters in vector `p`. These are the results shown in [Section 4](#4-results).

```
"test_metrics": {
  "r2": 0.9559590920924331,
  "nmse": 0.04404090592149448,
  "nrmse": 0.20985925815702633,
  "nmae": 0.07607935409954629
}
```

**Model with no CAD parameters**

![alt text](images/model_no_cadp.png)

This variant was studied as an alternative to the baseline. It trains without CAD parameters in vector `p`, representing scenarios where original CAD sketches are unavailable—a common occurrence when design history is lost or files are transferred between CAD systems.

```
"test_metrics": {
  "r2": 0.9045070139578009,
  "nmse": 0.09549298173583969,
  "nrmse": 0.3090193954276475,
  "nmae": 0.11722269859289076
}
```

Despite missing explicit CAD parameters, the shape embedder extracts geometric information from the STL file, achieving reasonable accuracy (though lower than the baseline).

![alt text](images/latent.png)

Analysis of the latent space reveals what geometric features the embedder captures:

- **Middle baffle plate angle (d61)** shows strong correlation with principal latent dimensions, indicating effective capture by the embedding architecture
- **Angle between ducts (d13)** shows partial representation in latent space
- **Inlet aperture length (d72)** and **curvature angle (d34)** have limited presence in the latent representation

Note that the shape embedder is trained jointly with the QoI regressor. Even though the embedder architecture only processes point clouds, training is end-to-end: the embedding learns to extract geometric features that, when combined with vector `p`, predict QoI effectively.

> 📝 **Note:** This architecture represents an initial baseline rather than an optimized solution. The primary objective is demonstrating end-to-end workflow feasibility. Future work can evaluate alternative embedding techniques (voxelization, signed distance fields, graph neural networks), different feature fusion methods for integrating `z` and `p`, or alternative predictors beyond Random Forests.
