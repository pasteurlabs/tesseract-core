# Ansys <-> Tesseract: Exploring QoI Workflows

## Table of Contents

[1. Case Study: HVAC Duct Dataset](#1-case-study-hvac-duct-dataset)
   - [1.1. Dataset Variations](#11-dataset-variations)
   - [1.2. QoI](#12-qoi)
   - [1.3. Dataset Summary](#13-dataset-summary)

[2. QoI Workflows](#2-qoi-workflows)
   - [2.1. CAD Geometry + Boundary Conditions → QoI Workflow](#21-cad-geometry--boundary-conditions---qoi-workflow)
   - [2.2. CAD Geometry + Boundary Conditions → Full-Field → QoI Workflow](#22-cad-geometry--boundary-conditions---full-field---qoi-workflow)

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

Engineers iterate over their designs to meet specific requirements, which are metrics that translate directly into product performance or business needs. We call these metrics Quantities of Interest (QoI). In many engineering contexts, engineers extract QoI as post-processed metrics from simulation solutions. For example, CFD simulations of a wing compute lift and drag coefficients as QoI. Engineering workflows involve iterative updates over design variables (like CAD parameters and boundary conditions) to achieve the required design performance. 

This showcase demonstrates how Tesseract and Ansys Fluent work together to extract insights from datasets of numerical simulations with QoI-based surrogacy. In particular, we will cover:
1. How to assemble an internal aerodynamics dataset, based on Ansys simulations, mapping CAD parameters and boundary conditions to QoI. 
2. How to create and analyze QoI-based surrogate models built on top of Ansys datasets, eliminating the need for time-consuming simulations
3. How to modularize and execute QoI-based workflows with Tesseracts that integrate with Ansys Fluent simulations outputs

## 1. Case Study: HVAC Duct Dataset
Prior to defining a QoI-based workflow, we created an internal aerodynamics dataset of ~300 samples with Ansys Fluent. This use case consisted of numerical simulations of internal air flow through an HVAC duct with two inner baffle plates. As the image below shows:

![alt text](images/duct_1.png)
- One baffle place is located just after the inlet acting as a flow constrainer
- The second baffle plate is in the intersection between duct branches and redirects the inlet flow

### 1.1 Dataset Variations

**CAD parameters**

Four geometric parameters were modified to explore different duct configurations:
- `d61`: Angle of the middle baffle plate
- `d72`: Aperture length of the inlet baffle plate
- `d13`: Angle between duct branches
- `d34`: Curvature angle of the duct branch

<table> <tr> <td align="center"> <img src="images/d13.png" alt="d13" width="400"/><br/> <b>d13:</b> Angle between duct branches </td> <td align="center"> <img src="images/d34.png" alt="d34" width="400"/><br/> <b>d34:</b> Curvature angle of duct branch </td> </tr> <tr> <td align="center"> <img src="images/d61.png" alt="d61" width="400"/><br/> <b>d61:</b> Angle of middle baffle plate </td> <td align="center"> <img src="images/d72.png" alt="d72" width="400"/><br/> <b>d72:</b> Aperture length of inlet baffle </td> </tr> </table>

**Boundary conditions**

The inlet velocity magnitude has been varied across numerical simulations to capture different flow regimes and operating conditions.

### 1.2. QoI
Although the dataset of numerical simulations was generated without a specific QoI-based workflow in mind, a text file containing static pressure values averaged at certain slices of the HVAC was also available as part of the results. These static pressure values will be considered as the base QoI of each numerical simulation. While the implemented code also allows to define additional metrics based on these base QoI (such as pressure drops $\Delta p$ across different slices), this showcase focuses only on the base static pressure values. 

In particular, the reported static pressure values are averaged at 4 different sections: `inlet`, `outlet`, `p2-plane` (YZ plane after inlet baffle plane) and `p3-plane` (YZ plane before middle baffle plane).

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
Understanding how geometry design impacts QoI is of valuable interest for CAD and simulation engineers. Establishing a workflow that directly maps a CAD file (e.g. .stl) to QoI removes the need for meshing, simulation, and post-processing. This enables a significant reduction in engineering time and accelerates design iteration.

In [the following section](#3-ansys-tesseract-qoi-workflow-proposal) we will outline how Tesseract allows us to define a workflow on top of Ansys Fluent simulations and discuss the benefits it provides. But first, let's explore some of the possibilities to define a QoI-based workflow.

### 2.1. CAD Geometry + Boundary Conditions -> QoI Workflow
<p align="center">
  <img src="images/qoi_overview.png">
</p>

This workflow enables engineers to directly predict QoI from CAD geometry files (e.g. stl) and boundary conditions. The workflow illustrated above has been selected for this showcase as it integrates methods that can be deployed during early-stage design and do not require complex full-field surrogate models. This ensures that the proposed solution remains practical, maintainable, and aligned with typical engineering development cycles.

### 2.2. CAD Geometry + Boundary Conditions -> Full-Field -> QoI Workflow
<p align="center">
  <img src="images/qoi_overview_full.png">
</p>

Engineers can also incorporate a full-field surrogate model into an alternative QoI-based workflow alongside direct CAD-to-QoI predictions. When a surrogate model generates full-field outputs (e.g., pressure distributions, temperature fields, stress contours), different QoI can be directly derived from these fields using standard post-processing tools. However, deploying a full-field surrogate model introduces additional complexity and may not deliver significant benefits for early-stage design studies.

## 3. Ansys<->Tesseract QoI Workflow Proposal
We can use Tesseract to define modular and reusable components that orchestrate the QoI-based workflows on top of the Ansys Fluent simulations based on HVAC systems. The diagram below shows an approach to perform training and inference on a QoI-based surrogate model that maps CAD geometries and boundary conditions to QoI  (see [reference](#21-cad-geometry--boundary-conditions---qoi-workflow)) with Tesseract.

![alt text](images/tesseract_wf.png)

To enable QoI-based predictions directly from CAD geometries and boundary conditions, we first need to train a surrogate model. With Tesseract, we can define two complementary workflows: a training workflow where the surrogate model learns from historical Ansys Fluent simulation data, and an inference workflow where the surrogate model predicts QoI for new CAD parameters and boundary conditions.

### 3.1. Tesseract Workflows and Components
#### 3.1.1 Tesseract Components
**Dataset Tesseract**
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
        description="List of npz files containing point-cloud data, simulation parameters and/or QoIs",
    )
```
**Training Tesseract**
```python
class InputSchema(BaseModel):
    config: InputFileReference = Field(description="Configuration file")
    data: list[str] = Field(
        description="List of npz file paths (can be absolute paths from dependent workflows)"
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
```python
class InputSchema(BaseModel):
    config: InputFileReference = Field(description="Configuration file")
    data: list[str | Path] = Field(
        description="List of npz files containing point-cloud data, simulation parameters and/or QoIs"
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
The training workflow consists of two key Tesseract components: 
- **Dataset Pre-Processing Tesseract**: Extracts and formats data from Ansys Fluent simulation runs. This component samples different geometries using point clouds and extracts the critical information downstream components need: CAD parameters, boundary conditions and QoI from each numerical simulation. 
- **Training Tesseract**: Consumes the processed dataset and defines a training loop to tune the QoI-based surrogate model, learning the functional mapping between HVAC geometries, boundary conditions, and simulated QoI.
#### 3.1.3. Tesseract Inference Workflow
The inference workflow enables engineers to predict QoI using only CAD geometries and boundary conditions.

This workflow consists of:
- **Dataset Pre-Processing Tesseract**: Ensures format compatibility with the trained model.
- **Inference Tesseract**: Feeds the pre-processed data into the trained model to predict QoI.

## 4. Results
Before looking at the results, a complete section on how the Ansys Fluent simulations are transformed into a dataset able to be fed to the QoI-based surrogate model is described in Appendix A. A brief summary is defined below:




## 5. Outlook
## Appendix
### A. QoI-based Surrogate Model
#### A.1. Dataset Preparation
Referring back to the HVAC dataset introduced earlier, for each of the Ansys simulation runs the following files are present: a CAD geometry file (`duct_baffle.stl`), mesh discretization (`duct_baffle.msh.h5`), Ansys Fluent solver case file (`duct_baffle.cas.h5`) and results data file (`duct_baffle.dat.h5`), static pressure report (`all_pressure.txt`) and a file containing the CAD and BC variations performed (`metadata.json.series`).
```
Experiment_0
|-duct_baffle.stl
|-duct_baffle.msh.h5
|-duct_baffle.cas.h5
|-duct_baffle.dat.h5
|-metadata.json.series
|-all_pressure.txt
```
A critical part for training a QoI-based surrogate model is the extraction of features from the available simulations. The following subsections describe the feature-extraction process in detail.
##### CAD Pre-Processing
Although several preprocessing strategies can be used to convert CAD geometry files into ML-ready inputs (e.g. voxelization, signed distance fields, graph-based representations…) this showcase adopts a point-cloud representation as it offers a good balance between geometric fidelity and implementation simplicity. 

*Point-Cloud Sampling (Points & Normals)*

Point-cloud sampling retrieves both point coordinates and their normals.

Similar to the Ansys Meshing feature “Sphere of Influence” (referred to as SoI in the images), the implemented code is able to increase the sampling density in targeted regions. Since the baffle plates represent a relevant area of the CAD design, spheres of influence around these areas can be defined. 
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

The `config.yaml` contains the parameters used to control the pre-processing of CAD geometry files, including the configuration of different point-cloud sampling techniques.

*Point-Cloud Derived Parameters*

The point-cloud representation also enables the extraction of several geometric metrics that are not explicitly available from the STL file but can be directly computed from the sampled points. These features provide global descriptors of the geometry and will be used to help the QoI-based surrogate model capture coarser attributes that could influence the QoI. The metrics considered include:

- min(x, y, z): minimum coordinate values across all sampled points

- max(x, y, z): maximum coordinate values across all sampled points

- size: bounding-box dimensions, computed as max(xyz) – min(xyz)

- diag: diagonal length of the bounding box, i.e., ‖size‖

- max_side: maximum extent among the three bounding-box dimensions

- centroid: geometric center of the sampled point cloud

*Sketch CAD parameters*

In addition to the CAD geometry file (STL), each Ansys simulation folder includes a file that records the key variations applied during the dataset generation. This `metadata.json.series` file was automatically produced and serves to keep track of the CAD design parameters and other variations performed.

```json
"variations": {
    "d61": 0.8,
    "d13": 1.7,
    "d72": 0.6,
    "d34": 2.7,
    ...
}
```

*Boundary conditions*

In the `metadata.json.series` file, the variations associated to the boundary conditions are also present. The pre-processing step for the boundary conditions simply extracts the values from the file.

```json
"variations": {
    ...
    "mean_velocity": 78.0 
}
```

*QoI*

Ansys Fluent simulation reports are generated for each simulation run and contain the static pressure values extracted at several stations (as described in [this section](#12-qoi)). 
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

A dedicated `SurfaceIntegralReport` class has been implemented to read these files and parse the pressure data.

#### A.2. Model Architecture