#!/usr/bin/env python3
import subprocess
import sys
import os


def run_seed(seed, object_path):
    gripper_path = "gripper_parameter/single_circle_cup_1cm_diam.yaml"

    object_name = os.path.splitext(os.path.basename(object_path))[0]
    output_dir = f"results/{object_name}_repeat_test"

    cmd = [
        sys.executable,
        "-u",
        "run_grasp_sampling.py",
        object_path,
        gripper_path,
        output_dir,
        str(seed),
        "--scale", "0.001"
    ]

    print(f"[Object {object_name}][Seed {seed}] Starting", flush=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(f"[Object {object_name}][Seed {seed}] {line}", end="", flush=True)

    process.wait()

    return seed, process.returncode


def main():
    object_dir = "Test_part"

    objects = [
        os.path.join(object_dir, f)
        for f in os.listdir(object_dir)
        if f.lower().endswith((".stl", ".obj", ".ply", ".pcd"))
    ]

    seeds = range(1, 2)

    print(f"Found {len(objects)} objects:")
    for obj in objects:
        print(f"  - {obj}")

    for object_path in objects:
        object_name = os.path.basename(object_path)
        print(f"\n========== Processing {object_name} ==========\n")

        for seed in seeds:
            seed, code = run_seed(seed, object_path)

            print(
                f"\n--- Object {object_name} Seed {seed} Finished with Exit Code {code} ---\n",
                flush=True
            )


if __name__ == "__main__":
    main()