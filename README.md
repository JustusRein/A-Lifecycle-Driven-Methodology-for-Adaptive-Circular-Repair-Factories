## Project Overview
This repository contains the advanced modular framework for robotic grasping of components-

While the legacy system focused on parallel-jaw grippers, this version introduces a high-fidelity **Vacuum Grasping Pipeline** specifically designed for industrial servicing of assembly parts.

---

## Key Features
- **Intelligent Plane Alignment**: Uses a hybrid **RANSAC + DBSCAN** approach to identify solid, contiguous surfaces for grasping, ignoring sparse or noisy regions.
- **Grasp Stability Score (GSS)**: A physics-informed evaluation engine that simulates suction seal quality by projecting pad geometries onto the object surface and analyzing zone-based point density.
- **Multi-Pad Support**: Configure complex suction arrays (circular cups or rectangular foam pads) via YAML.
- **Modular Sampler (v2)**: Decoupled candidate generation (Raycasting) from evaluation strategies.
- **Research Playground**: A structured Jupyter environment for rapid experimentation with new parts and grippers.

---

## Repository Structure
- `src/grasping/` – Core sampling logic and GSS evaluation.
- `src/grippers/` – Physical models for vacuum grippers and suction pads.
- `src/utils/` – Geometric math engine (Rodrigues rotations, diversity filters, etc.).
- `config/` – System-wide sampling and scoring parameters.
- `gripper_parameter/` – Library of suction cup configurations (Franka, Schmalz, etc.).
- `Test_part/Vulkan/` – STL/PLY library of satellite components.
- `workflow_vacuum_vulkan_playground.ipynb` – **Primary Entry Point** .

---

## Getting Started

### 1. Installation
Ensure you have [Conda](https://docs.conda.io/) installed, then run "Anaconda Promt" and navigate to the project folder:
```bash
conda env create -f environment.yml (for linux)
conda env create -f environment_windows.yml (for windows)
conda activate <env_name>
```

venv in windows without yml
- "py -3.12 -m venv .venv"
- ".venv/Scripts/activate"
- "pip install -r requirements.txt"

### 2. Running the Pipeline
The most intuitive way to use this framework is through the **Playground**:
1. Open `jupyter lab`
2. Open `workflow_vacuum_vulkan_playground.ipynb` in jupyter lab.

### If you want to change code and see a green rectangle as a cursor press i to enter insert mode to write in the cells(happens with jupiterlab-vim).

3. Select your gripper (e.g., `double_cup_1cm_1cm.yaml`).
4. Select your Vulkan part (e.g., `part_1.stl`).
5. Run the cells to visualize alignment, heatmaps, and optimal grasp candidates.



### 3. Custom Configurations
- Use `config/config_template.yaml` to adjust sampling density and scoring weights.
- Use `gripper_parameter/vacuum_gripper_template.yaml` to design your own custom suction arrays.

---

## Analytics & Debugging
The framework provides deep insights into why a grasp succeeds or fails:
- **Heatmaps**: Spatial visualization of sealing, torque, and verticality scores.
- **2D Projections**: Visual inspection of exactly how a suction cup "sees" the surface under it.
- **Collision Shields**: 3D visualization of pre-grasp approach volumes.

---

## License
This code is released under the **MIT License**. It is intended for research and educational purposes but can be used for any other usecase.

## Contact
**Justus Rein** - [rein@plcm.tu-darmstadt.de](mailto:rein@plcm.tu-darmstadt.de)  
*Product Life Cycle Management (PLCM) - TU Darmstadt*
