# 🌌 Adaptive In-Orbit Servicing: Vacuum Grasping Framework

<a href="https://doi.org/10.5281/zenodo.16947438 "><img src="https://zenodo.org/badge/1038437832.svg" alt="DOI"></a>

## 🚀 Project Overview
This repository contains the advanced modular framework for robotic grasping of complex satellite components, accompanying the research paper:

**"Adaptive In-Orbit Servicing of Altered Satellite Components"**  
*Authors: Justus Rein, Christian Plesker, Adrian Reuther, Hanyu Liu, Benjamin Schleich*

While the legacy system focused on parallel-jaw grippers, this version introduces a high-fidelity **Vacuum Grasping Pipeline** specifically designed for industrial servicing of assembly parts.

---

## ✨ Key Features
- **Intelligent Plane Alignment**: Uses a hybrid **RANSAC + DBSCAN** approach to identify solid, contiguous surfaces for grasping, ignoring sparse or noisy regions.
- **Grasp Stability Score (GSS)**: A physics-informed evaluation engine that simulates suction seal quality by projecting pad geometries onto the object surface and analyzing zone-based point density.
- **Multi-Pad Support**: Configure complex suction arrays (circular cups or rectangular foam pads) via YAML.
- **Modular Sampler (v2)**: Decoupled candidate generation (Raycasting) from evaluation strategies.
- **Research Playground**: A structured Jupyter environment for rapid experimentation with new parts and grippers.

---

## 📂 Repository Structure
- `src/grasping/` – Core sampling logic and GSS evaluation.
- `src/grippers/` – Physical models for vacuum grippers and suction pads.
- `src/utils/` – Geometric math engine (Rodrigues rotations, diversity filters, etc.).
- `config/` – System-wide sampling and scoring parameters.
- `gripper_parameter/` – Library of suction cup configurations (Franka, Schmalz, etc.).
- `Test_part/Vulkan/` – STL/PLY library of satellite components.
- `workflow_vacuum_vulkan_playground.ipynb` – **Primary Entry Point** .

---

## 🛠️ Getting Started

### 1. Installation
Ensure you have [Conda](https://docs.conda.io/) installed, then run:
```bash
conda env create -f environment.yml
conda activate <env_name>
```

### 2. Running the Pipeline
The most intuitive way to use this framework is through the **Playground**:
1. Open `workflow_vacuum_vulkan_playground.ipynb`.
2. Select your gripper (e.g., `double_cup_1cm_1cm.yaml`).
3. Select your Vulkan part (e.g., `part_1.stl`).
4. Run the cells to visualize alignment, heatmaps, and optimal grasp candidates.

### 3. Custom Configurations
- Use `config/config_template.yaml` to adjust sampling density and scoring weights.
- Use `gripper_parameter/vacuum_gripper_template.yaml` to design your own custom suction arrays.

---

## 📊 Analytics & Debugging
The framework provides deep insights into why a grasp succeeds or fails:
- **Heatmaps**: Spatial visualization of sealing, torque, and verticality scores.
- **2D Projections**: Visual inspection of exactly how a suction cup "sees" the surface under it.
- **Collision Shields**: 3D visualization of pre-grasp approach volumes.

---

## 📄 License
This code is released under the **MIT License**. It is intended for research and educational purposes.

## ✉️ Contact
**Justus Rein** - [rein@plcm.tu-darmstadt.de](mailto:rein@plcm.tu-darmstadt.de)  
*Product Life Cycle Management (PLCM) - TU Darmstadt*
