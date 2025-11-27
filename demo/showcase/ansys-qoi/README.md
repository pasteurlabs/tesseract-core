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

### 1.2. QoI
Although the dataset simulations were generated before thinking of any QoI workflow, a text file containing static pressure values averaged at certain stations was dumped as part of the reporting. This static pressure values will be considered as the base QoI of the simulation, although we will also explore how to define additional metrics based on these QoI (e.g. $\Delta p$). 

The static pressure values reported are averaged at 4 different sections: `inlet`, `outlet`, `p2-plane` (YZ plane after inlet baffle plane) and `p3-plane` (YZ plane before middle baffle plane)

<table> <tr> <td align="center"> <img src="images/p2-plane.png" alt="p2-plane" height="300"/><br/> <b>p2-plane:</b> YZ plane after inlet baffle plate </td> <td align="center"> <img src="images/p3-plane.png" alt="p3-plane" height="300"/><br/> <b>p3-plane:</b> YZ plane before middle baffle plate </td> </tr> </table>

> 📝 **Note:** How Ansys reporting tool enables the extraction of QoI 
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
**Boundary Condition Variations**

The team also varied the inlet velocity magnitude across simulations to capture different flow regimes and operating conditions.

## QoI Workflows

