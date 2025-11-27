# Ansys <-> Tesseract: Exploring QoI Workflows

Engineers iterate over their designs to meet specific requirements, which are metrics that translate directly into product performance or business needs. We call these metrics Quantities of Interest (QoI). In many engineering contexts, engineers extract QoI as post-processed metrics from simulation solutions. For example, CFD simulations of a wing compute lift and drag coefficients as QoI. Engineering workflows involve iterative updates over design variables (like CAD parameters and boundary conditions) to achieve the required design performance. 

This showcase demonstrates how Tesseract and Ansys Fluent work together to extract insights from QoI datasets. In particular, we will cover:
1. How to assemble an internal aerodynamics dataset with QoI metrics, based on Ansys simulations
2. How to create and analyze QoI surrogate machine learning models built on top of Ansys datasets, eliminating the need for time-consuming simulations
3. How to modularize and execute QoI workflows with Tesseracts that integrate with Ansys simulations outputs

## 1. Case Study: HVAC Duct Dataset
Prior to defining a QoI workflow, we created an internal aerodynamics dataset of ~300 samples with Ansys Fluent. This use case consisted of simulations of air flowing through an HVAC duct with two innter baffle plates. As the image below shows:

![alt text](images/duct_1.png)
- One baffle place is located just after the inlet acting as a flow constrainer
- The second baffle plate is in the intersection between duct branches and redirects the inlet flow

### 1.1 Dataset Variations

**CAD Variations**

Four geometric parameters were modified to explore different duct configurations:
- `d61`: Angle of the middle baffle plate
- `d72`: Aperture length of the inlet baffle plate
- `d13`: Angle between duct branches
- `d34`: Curvature angle of the duct branch

<table> <tr> <td align="center"> <img src="images/d13.png" alt="d13" width="400"/><br/> <b>d13:</b> Angle between duct branches </td> <td align="center"> <img src="images/d34.png" alt="d34" width="400"/><br/> <b>d34:</b> Curvature angle of duct branch </td> </tr> <tr> <td align="center"> <img src="images/d61.png" alt="d61" width="400"/><br/> <b>d61:</b> Angle of middle baffle plate </td> <td align="center"> <img src="images/d72.png" alt="d72" width="400"/><br/> <b>d72:</b> Aperture length of inlet baffle </td> </tr> </table>

**Boundary Condition Variations**

The team also varied the inlet velocity magnitude across simulations to capture different flow regimes and operating conditions.

### 1.2. QoI
Although the dataset simulations were generated before thinking of any QoI workflow, a text file containing static pressure values averaged at certain stations was dumped as part of the reporting. This static pressure values will be considered as the base QoI of the simulation, although we will also explore how to define additional metrics based on these QoI (e.g. $\Delta p$). 

The static pressure values reported are averaged at 4 different sections: `inlet`, `outlet`, `p2-plane` (YZ plane after inlet baffle plane) and `p3-plane` (YZ plane before middle baffle plane)

<table> <tr> <td align="center"> <img src="images/p2-plane.png" alt="p2-plane" height="300"/><br/> <b>p2-plane:</b> YZ plane after inlet baffle plate </td> <td align="center"> <img src="images/p3-plane.png" alt="p3-plane" height="300"/><br/> <b>p3-plane:</b> YZ plane before middle baffle plate </td> </tr> </table>

> 📝 **Note:** How Ansys Fluent reporting tool enables the extraction of QoI 
`/report/surface-integrals area-weighted-avg inlet outlet p2-plane p3-plane () pressure yes all_pressure.txt`

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
Understanding how geometry design impacts QoI is of valuable interest for CAD and simulation engineers. Establishing a workflow that directly maps a CAD file (e.g. .stl) to QoI removes the need for meshing, simulation, and post-processing. This enables significant reductions in engineering time and accelerates design iteration.

In [the following section](#3-ansys-tesseract-qoi-workflow-proposal) we will outline how Tesseracts are able to define a workflow on top of Ansys simulations and mention the benefits it provides. But first, let's explore some of the possibilities to define QoI workflows.

### 2.1. CAD Geometry + Boundary Conditions -> QoI Workflow
<p align="center">
  <img src="images/qoi_overview.png">
</p>

This workflow enables engineers to directly predict QoI from CAD geometry and boundary conditions. The workflow illustrated above has been selected for this demonstration as it integrates methods that can be deployed during early-stage design and do not require highly complex neural network architectures. This ensures that the proposed solutions remain practical, maintainable, and aligned with typical engineering development cycles.

### 2.2. CAD Geometry + Boundary Conditions -> Full-Field -> QoI Workflow
<p align="center">
  <img src="images/qoi_overview_full.png">
</p>

Engineers can also incorporate a full-field surrogate model into an alternative QoI workflow alongside direct CAD-to-QoI prediction. When a surrogate predictor generates full-field outputs (e.g., pressure distribution, temperature fields, stress contours), the QoI can be directly derived from these fields using standard post-processing tools. However, deploying a full-field surrogate introduces additional complexity and may not deliver significant benefits for early-stage design studies.

## 3. Ansys<->Tesseract QoI Workflow Proposal
We can use Tesseracts to define modular and reusable components that orchestrate the QoI workflow on top of the Ansys HVAC simulations. The diagram below shows an approach to perform training an inference on a QoI surrogate model that maps CAD geometry files and boundary conditions to QoI  (see [reference](#21-cad-geometry--boundary-conditions---qoi-workflow)) with Tesseracts.

![alt text](images/tesseract_wf.png)

To enable QoI prediction directly from CAD files and boundary conditions, engineers must first train a model. With this Tesseract architecture we can define two complementary workflows, a training workflow that learns from historical Ansys simulations, and an inference workflow that predicts QoI for new CAD designs.

### 3.1. Tesseract Workflows and Components
#### 3.1.1 Training Workflow
The training workflow consists of two key Tesseract components: 
- **Dataset Pre-Processing Tesseract**: Extracts and formats data from Ansys simulation runs. This component samples geometry (point clouds in this case) and extracts the critical information downstream components need: boundary conditions and QoI values from each simulation. 
- **Training Tesseract**: Consumes the processed dataset and defines a training loop to tune the QoI model's weights, learning the relationship between geometry, boundary conditions, and performance outcomes.
#### 3.1.2. Inference Workflow
The inference workflow enables engineers to predict QoI using only CAD geometry and boundary condition parameters.

This workflow consists of:
- **Dataset Pre-Processing Tesseract**: Ensures format compatibility with the trained model.
- **Inference Tesseract**: Feeds the pre-processed data into the trained model to predict QoI.
#### 3.1.3. Tesseract Components
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
#### 3.1.4. Putting them all together
### 3.2. But... Why use Tesseracts?
## 4. Results
## 5. Outlook
## A. Annex - QoI Surrogate Model
### A.1. Dataset Preparation
### A.2. QoI Model Architecture