# Adaptive In-Orbit Servicing of Altered Satellite Components

<a href="https://doi.org/10.5281/zenodo.16947438 "><img src="https://zenodo.org/badge/1038437832.svg" alt="DOI"></a>

## Project Overview
This repository contains the code accompanying the paper:

**"Adaptive In-Orbit Servicing of Altered Satellite Components"**  
*Authors: Justus Rein, Christian Plesker, Adrian Reuther, Hanyu Liu, Benjamin Schleich*  

The code is intended to reproduce the experiments, figures, and results presented in the publication.  

## Repository Structure
- "Test_part/Verification_examples/" – Input datasets and Output figures
- "gripper_paramters/" – Parameter set for the used grippers and robot arms
- "environment.yml" – Configuration file
- "main_script.py" – Main script  
- "record_excute_time.py" – records the runtime of the main_script.py
- "step2clustered_pcd.py" – converts a step file to a clustered point cloud with one cluster per part in the assembly (currently hardcoded because the location and rotation from step file is not imported)    

## Requirements
- Dependencies: listed in "environment.yml"  

## Usage
1. Create the conda environment from the provided `environment.yml` file: 
conda env create -f environment.yml
2. Activate the environment: 
conda activate <myproject>
2.1. If required a clustered pcd can be generated using the "step2clustered_pcd.py":
python step2clustered_pcd.py
3. Run the main script: 
python main_script.py

## Data Availability
A subset of the dataset is provided under "Test_part/Verification_examples/" for demonstration purposes.

## Citation
If you use this repository, please cite the paper:

Citation information will be updated once the paper is accepted and published.

## License
This code is released under the MIT license.
It is intended for research and educational purposes in the context of the above publication but can also be used for any other use case.

## Contact
Justus Rein - rein@plcm.tu-darmstadt.de