#!/usr/bin/env python3
import subprocess
from multiprocessing import Pool
import os

import sys

def run_seed(seed):
    object_path = "Test_part/cubesat.stl"
    gripper_path = "gripper_parameter/single_circle_cup_1cm_diam.yaml"
    output_dir = "results/cubesat_repeat_test"
    
    cmd = [
        sys.executable, "run_grasp_sampling.py",
        object_path,
        gripper_path,
        output_dir,
        str(seed),
        "--scale", "0.001"
    ]
    # We do NOT pass --align as requested ("align should be false by default")
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return seed, result.returncode, result.stdout, result.stderr

def main():
    seeds = list(range(1, 51))
    print("Starting parallel execution for seeds 1 to 50...")
    
    with Pool() as pool:
        results = pool.map(run_seed, seeds)
        
    for seed, code, stdout, stderr in results:
        print(f"\n--- Seed {seed} Finished with Exit Code {code} ---")
        if stdout:
            print("STDOUT:")
            print(stdout.strip())
        if stderr:
            print("STDERR:")
            print(stderr.strip())

if __name__ == "__main__":
    main()
