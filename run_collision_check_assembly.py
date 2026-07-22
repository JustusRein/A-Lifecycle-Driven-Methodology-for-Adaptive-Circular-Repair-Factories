#!/usr/bin/env python3

import subprocess
import sys
import os


def run_seed(seed, object_cloud, obstacle_cloud):

    object_cloud = os.path.join(
        "input_collision",
        object_cloud
    )

    if obstacle_cloud is not None:
        obstacle_cloud = os.path.join(
            "input_collision",
            obstacle_cloud
        )

    gripper_path = "gripper_parameter/single_circle_cup_1cm_diam.yaml"

    object_name = os.path.splitext(os.path.basename(object_cloud))[0]

    obstacle_name = os.path.splitext(
        os.path.basename(obstacle_cloud)
    )[0]

    output_dir = f"results/{object_name}_vs_{obstacle_name}_seed_{seed}"

    cmd = [
        sys.executable,
        "-u",
        "run_grasp_sampling.py",
        object_cloud,
        gripper_path,
        output_dir,
        str(seed),
        "--obstacle",
        obstacle_cloud,
        "--scale",
        "0.001"
    ]

    print("\n==============================")
    print(f"Object:   {object_cloud}")
    print(f"Obstacle: {obstacle_cloud}")
    print(f"Seed:     {seed}")
    print("==============================")

    subprocess.run(cmd, check=True)


def main():

    test_cases = [

        # 0,1,2,3 greifen in 4,5
        ("unknown_0__unknown_1__unknown_2__unknown_3.pcd",
         "unknown_4__unknown_5.pcd"),

        # 4,5 greifen in 0,1,2,3
        ("unknown_4__unknown_5.pcd",
         "unknown_0__unknown_1__unknown_2__unknown_3.pcd"),

        # 0,2,3,4,5 greifen in 1
        ("unknown_0__unknown_2__unknown_3__unknown_4__unknown_5.pcd",
         "unknown_1.pcd"),

        # 1 greifen in 0,2,3,4,5
        ("unknown_1.pcd",
         "unknown_0__unknown_2__unknown_3__unknown_4__unknown_5.pcd"),

        # 4 greifen in 5
        ("unknown_4.pcd",
         "unknown_5.pcd"),

        # 5 greifen in 4
        ("unknown_5.pcd",
         "unknown_4.pcd"),

        # 0 greifen in 1,2,3
        ("unknown_0.pcd",
         "unknown_1__unknown_2__unknown_3.pcd"),

        # 1,2,3 greifen in 0
        ("unknown_1__unknown_2__unknown_3.pcd",
         "unknown_0.pcd"),

        # 1 greifen in 0,2,3
        ("unknown_1.pcd",
         "unknown_0__unknown_2__unknown_3.pcd"),

        # 0,2,3 greifen in 1
        ("unknown_0__unknown_2__unknown_3.pcd",
         "unknown_1.pcd"),

        # 0,2,3 greifen in 4,5
        ("unknown_0__unknown_2__unknown_3.pcd",
         "unknown_4__unknown_5.pcd"),

        # 4,5 greifen in 0,2,3
        ("unknown_4__unknown_5.pcd",
         "unknown_0__unknown_2__unknown_3.pcd"),

        # 1 greifen in 2,3
        ("unknown_1.pcd",
         "unknown_2__unknown_3.pcd"),

        # 2,3 greifen in 1
        ("unknown_2__unknown_3.pcd",
         "unknown_1.pcd"),

        # 2 greifen in 0,3
        ("unknown_2.pcd",
         "unknown_0__unknown_3.pcd"),

        # 0,3 greifen in 2
        ("unknown_0__unknown_3.pcd",
         "unknown_2.pcd"),

        # 0 greifen in 2,3
        ("unknown_0.pcd",
         "unknown_2__unknown_3.pcd"),

        # 2,3 greifen in 0
        ("unknown_2__unknown_3.pcd",
         "unknown_0.pcd"),

        # 2 greifen in 3
        ("unknown_2.pcd",
         "unknown_3.pcd"),

        # 3 greifen in 2
        ("unknown_3.pcd",
         "unknown_2.pcd"),

        # 0 greifen in 3
        ("unknown_0.pcd",
         "unknown_3.pcd"),

        # 3 greifen in 0
        ("unknown_3.pcd",
         "unknown_0.pcd"),


        # Einzelteile ohne Hindernisse
        ("unknown_0.pcd", None),
        ("unknown_1.pcd", None),
        ("unknown_2.pcd", None),
        ("unknown_3.pcd", None),
        ("unknown_4.pcd", None),
        ("unknown_5.pcd", None),

        # Kombinationen ohne Hindernisse
        ("unknown_0__unknown_1__unknown_2__unknown_3.pcd", None),
        ("unknown_0__unknown_1__unknown_2__unknown_3__unknown_4__unknown_5.pcd", None),
        ("unknown_0__unknown_2__unknown_3.pcd", None),
        ("unknown_0__unknown_2__unknown_3__unknown_4__unknown_5.pcd", None),
        ("unknown_0__unknown_3.pcd", None),
        ("unknown_1__unknown_2__unknown_3.pcd", None),
        ("unknown_2__unknown_3.pcd", None),
        ("unknown_4__unknown_5.pcd", None),
    ]

    seeds = range(1, 2)

    for object_cloud, obstacle_cloud in test_cases:
        for seed in seeds:
            run_seed(
                seed,
                object_cloud,
                obstacle_cloud
            )


if __name__ == "__main__":
    main()