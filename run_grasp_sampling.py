#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random
import yaml
import copy
import shutil
from os.path import join as pjoin
import numpy as np
import open3d as o3d

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom Framework Modules
from src.grippers.vacuum_gripper_v2 import VacuumGripper
from src.grasping.vacuum_sampler_v2 import VacuumGraspSampler, VacuumSamplerConfig
import src.utils.geometry_utils as gu
from src.generic_geometry import GenericGeometry

def convert_numpy(obj):
    """Recursively converts numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description="Run Grasp Sampling Pipeline in Headless Mode")
    parser.add_argument("object_path", type=str, help="Path to STL/PLY/PCD object file")
    parser.add_argument("gripper_path", type=str, help="Path to gripper configuration YAML")
    parser.add_argument("output_dir", type=str, help="Directory to save output results")
    parser.add_argument("seed", type=int, help="Random seed for NumPy, Python, and Open3D")
    parser.add_argument("--config_path", type=str, default=pjoin("config", "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--scale", type=float, default=1.0, help="Optional scaling factor for the object")
    parser.add_argument("--align", action="store_true", help="If set, aligns the largest plane to Z-axis using RANSAC")
    parser.add_argument("--ransac_iterations", type=int, default=5000, help="Number of iterations for RANSAC segmentation")
    parser.add_argument("--voxel_size", type=float, default=None, help="Optional downsample voxel size (defaults to suggested/2)")

    args = parser.parse_args()

    # 1. Set Random Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)

    print(f" Initializing Grasp Sampler script")
    print(f"   - Object: {args.object_path}")
    print(f"   - Gripper: {args.gripper_path}")
    print(f"   - Seed: {args.seed}")
    print(f"   - Align: {args.align} (RANSAC iterations: {args.ransac_iterations if args.align else 'N/A'})")

    # 2. Ensure Output Directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Load Configurations
    if not os.path.exists(args.config_path):
        print(f" Error: Config file not found at {args.config_path}")
        sys.exit(1)

    with open(args.config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Copy config to output directory if it doesn't exist yet
    dest_config = pjoin(args.output_dir, "config.yaml")
    if not os.path.exists(dest_config):
        shutil.copyfile(args.config_path, dest_config)

    sampler_config = VacuumSamplerConfig(**raw_config)

    # 4. Initialize Objects
    gripper = VacuumGripper(args.gripper_path)
    pcd = GenericGeometry(args.object_path)

    # 5. Optional Scale
    if args.scale != 1.0:
        print(f"   - Scaling object by factor: {args.scale}")
        pcd.scale(args.scale)

    # 6. Geometric Normalization & Downsample
    report = pcd.get_dimensions_report()
    voxel_size = args.voxel_size if args.voxel_size is not None else (report["suggested_voxel_size"] / 2.0)
    print(f"   - Voxel Size: {voxel_size:.6f} m")
    
    pcd_down = pcd.downsample(voxel_size=voxel_size)
    pcd_down = GenericGeometry(geometry=pcd_down)

    # 7. Alignment
    if args.align:
        print("   - Aligning largest plane to Z-axis (RANSAC)...")
        pcd_aligned_list = gu.align_largest_plane_to_z(
            pcd_down.geometry,
            distance_threshold=1.5 * voxel_size,
            n_iterations=args.ransac_iterations,
            k=3,
            theta_min_diff=30.0
        )
    else:
        pcd_aligned_list = [pcd_down.geometry]

    # 8. Grasp Sampling
    print("   - Starting sampling loop...")
    all_candidates = []
    
    for idx, aligned_geom in enumerate(pcd_aligned_list):
        sampler = VacuumGraspSampler(gripper, sampler_config)
        found_grasps = sampler.sample_grasps(aligned_geom)
        
        # Mark orientation index
        for g in found_grasps:
            g.origin_idx = idx
            
        all_candidates.extend(found_grasps)

    # Global Ranking
    all_candidates.sort(key=lambda x: x.score, reverse=True)
    num_candidates = len(all_candidates)
    print(f" Sampling complete. Found {num_candidates} valid candidates.")

    # 9. Format Results
    results = {
        "seed": args.seed,
        "object_path": args.object_path,
        "gripper_path": args.gripper_path,
        "num_candidates": num_candidates,
        "best_grasp": None,
        "all_candidates": []
    }

    # Format all candidates details
    for c in all_candidates:
        candidate_data = {
            "score": float(c.score),
            "position": c.transform[:3, 3].tolist(),
            "direction": c.approach_vector.tolist(),
            "transform": c.transform.tolist(),
            "origin_idx": int(c.origin_idx) if hasattr(c, "origin_idx") else 0,
            "score_details": convert_numpy(c.score_details)
        }
        results["all_candidates"].append(candidate_data)

    # Identify the best grasp
    if all_candidates:
        results["best_grasp"] = results["all_candidates"][0]
        best_pos = results["best_grasp"]["position"]
        best_score = results["best_grasp"]["score"]
        print(f" Best Grasp: Score {best_score:.4f} at Position {best_pos}")
    else:
        print(" No valid candidates found.")

    # 10. Write Results to JSON
    output_file = pjoin(args.output_dir, f"results_{args.seed}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f" Results saved successfully to {output_file}\n")

if __name__ == "__main__":
    main()
