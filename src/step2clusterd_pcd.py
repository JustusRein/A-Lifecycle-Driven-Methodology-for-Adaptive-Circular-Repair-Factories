import numpy as np
import open3d as o3d
import argparse
import os
import trimesh
import random
import copy

# Define components with position + rotation
components = {
    "01_Bottom_CubeSat_MODEL": {
        "position": [94.08, 150.34, 54.14],
        "rotation": [0.0, 0.0, 0.0, 1.0],
    },
    "03_Board_Empty": {
        "position": [44.08, 111.64, 98.74],
        "rotation": [180.0, 0.0, 0.71, 0.71],
    },
    "02_Wall_with_acor_logo": {
        "position": [93.08, 54.68, 70.14],
        "rotation": [-180.0, 0.0, 0.0, 1.0],
    },
    "02_Wall_no_Logo_1": {
        "position": [89.74, 52.34, 138.14],
        "rotation": [180.0, 0.71, 0.71, 0.0],
    },
    "02_wall_with_PLCM_logo_plcm": {
        "position": [92.08, 146.00, 138.14],
        "rotation": [180.0, 0.0, 1.0, 0.0],
    },
    "02_Wall_no_Logo": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 1.0, 0.0]},
    "PLCM_Logo": {"position": [32.24, -6.33, 66.28], "rotation": [0.0, 0.0, 1.0, 0.0]},
}

# Subassemblies: name → list of parts
subassemblies = {"02_wall_with_PLCM_logo_plcm": ["02_Wall_no_Logo", "PLCM_Logo"]}

# All children of subassemblies
components_in_subassemblies = set()
for comps in subassemblies.values():
    components_in_subassemblies.update(comps)


# Helper function: STEP → point clouds
def step_to_point_clouds(step_path, num_points=5000):
    scene = trimesh.load(step_path, file_type="step")
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError("❌ STEP could not be loaded as a scene.")
    if len(scene.geometry) == 0:
        raise RuntimeError("❌ No geometries found in STEP file.")

    pcs = {}
    for name, geom in scene.geometry.items():
        mesh = trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces)
        if mesh.is_empty:
            continue

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()

        pcd = o3d_mesh.sample_points_poisson_disk(num_points)
        pcs[name] = pcd
    return pcs


# Transformations
def create_transform_matrix(rotation, position):
    position = np.array(position, dtype=float) / 1000.0  # mm → m
    angle_deg, ax, ay, az = rotation
    angle = np.radians(angle_deg)
    axis = np.array([ax, ay, az], dtype=float)
    if np.linalg.norm(axis) < 1e-9:
        R = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c
        R = np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ]
        )
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = position
    return M


def apply_transform(pcd, rotation, position):
    M = create_transform_matrix(rotation, position)
    pcd.transform(M)
    return pcd


def apply_subassembly_transform(
    pcd, parent_rotation, parent_position, local_rotation, local_position
):
    M_local = create_transform_matrix(local_rotation, local_position)
    M_parent = create_transform_matrix(parent_rotation, parent_position)
    M_total = M_parent @ M_local
    pcd.transform(M_total)
    return pcd


# Helper function to save PLY + PCD
def save_point_clouds(pcd, path_base):
    o3d.io.write_point_cloud(path_base + ".ply", pcd)
    o3d.io.write_point_cloud(path_base + ".pcd", pcd)


# Manage cluster colors
cluster_colors = {}


def get_cluster_color(name):
    if name not in cluster_colors:
        cluster_colors[name] = [random.random(), random.random(), random.random()]
    return cluster_colors[name]


def main():
    parser = argparse.ArgumentParser(
        description="STEP → Point clouds with transformations"
    )
    parser.add_argument(
        "--input",
        default=os.path.join("Test_part", "assembly.step"),
        help="Input STEP file",
    )
    parser.add_argument(
        "--output_dir", default="output", help="Directory for point clouds"
    )
    parser.add_argument(
        "--num_points", type=int, default=5000, help="Number of sample points per part"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Converting STEP to individual point clouds…")
    try:
        pcs = step_to_point_clouds(args.input, args.num_points)
    except RuntimeError as e:
        print(e)
        return

    combined = o3d.geometry.PointCloud()
    geometries = []

    for comp_name, transform in components.items():
        # Subassembly
        if comp_name in subassemblies:
            for subpart in subassemblies[comp_name]:
                if subpart not in pcs:
                    continue
                pcd = copy.deepcopy(pcs[subpart])
                apply_subassembly_transform(
                    pcd,
                    parent_rotation=components[comp_name]["rotation"],
                    parent_position=components[comp_name]["position"],
                    local_rotation=components[subpart]["rotation"],
                    local_position=components[subpart]["position"],
                )
                color = np.tile(get_cluster_color(subpart), (len(pcd.points), 1))
                pcd.colors = o3d.utility.Vector3dVector(color)

                out_path_base = os.path.join(args.output_dir, f"{comp_name}_{subpart}")
                save_point_clouds(pcd, out_path_base)

                combined += pcd
                geometries.append(pcd)

        # Save single part (if not in subassembly)
        if comp_name in pcs and comp_name not in components_in_subassemblies:
            pcd = copy.deepcopy(pcs[comp_name])
            apply_transform(pcd, transform["rotation"], transform["position"])
            color = np.tile(get_cluster_color(comp_name), (len(pcd.points), 1))
            pcd.colors = o3d.utility.Vector3dVector(color)

            out_path_base = os.path.join(args.output_dir, f"{comp_name}")
            save_point_clouds(pcd, out_path_base)

            combined += pcd
            geometries.append(pcd)

    # Save combined point cloud
    out_path_base = os.path.join(args.output_dir, "combined")
    save_point_clouds(combined, out_path_base)

    print(f"👉 Combined point cloud saved at: {out_path_base}.ply / .pcd")
    print("Opening Open3D Viewer…")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Assembly Point Cloud (Cluster colors)",
        width=1280,
        height=720,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
