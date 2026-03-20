import yaml
import numpy as np
import open3d as o3d
import trimesh
from dataclasses import dataclass

from src.grippers.base_gripper import BaseGripper
from src.generic_geometry import GenericGeometry
import src.utils.geometry_utils as gu


@dataclass
class ParallelGripperConfig:
    """Dataclass mapping the Franka.yaml parameters (in meters)."""

    # Finger Dimensions
    a_pg: float  # Finger width (X)
    e_pg: float  # Finger depth (Y)
    b_pg: float  # Gripper contact area length (Z part 1)
    c_pg: float  # Gripper non-contact length (Z part 2)

    # Gripper Opening
    f_pg: float  # Max opening distance
    g_pg: float  # Min closing distance

    # Gripper Base (Bottom Part - Closer to fingers)
    h_pg: float  # Width (X)
    l_pg: float  # Depth (Y)
    t_pg: float  # Length (Z)

    # Gripper Base (Top Part - Closer to robot flange)
    q_pg: float  # Width (X)
    o_pg: float  # Depth (Y)
    u_pg: float  # Length (Z)

    # Robot Arm Link (Flange connection)
    ra: float  # Width (X)
    rb: float  # Depth (Y)
    rc: float  # Length (Z)

    # --- Safety / Clearance Parameters (Used by Sampler, not Visualizer) ---
    w_pg: float
    v_pg: float
    i_pg: float
    d_pg: float
    x_pg: float
    k_pg: float
    m_pg: float
    r_pg: float
    p_pg: float
    re: float
    rf: float
    rj: float
    z_pg: float


class ParallelGripper(BaseGripper):
    """
    Concrete implementation of a Parallel Gripper based on Franka parameters.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.config: ParallelGripperConfig = self.load_config()
        self.collision_geometry = self.generate_collision_mesh()
        print(
            f"🔧 Parallel Gripper initialized with max opening: {self.config.f_pg:.3f}m"
        )

    def load_config(self) -> ParallelGripperConfig:
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ParallelGripperConfig(**data)

    def generate_collision_mesh(self) -> GenericGeometry:
        """
        Generates a complete 3D representation of the Franka parallel gripper.
        Follows the Open3D paradigm established in the Vacuum Gripper to ensure
        perfect color retention during merges.
        """
        c = self.config
        meshes = []

        # Open3D paint_uniform_color expects RGB in 0.0 to 1.0 range
        color_finger = [112 / 255.0, 48 / 255.0, 160 / 255.0]  # Purple
        color_base_bot = [237 / 255.0, 125 / 255.0, 49 / 255.0]  # Orange
        color_base_top = [210 / 255.0, 100 / 255.0, 30 / 255.0]  # Darker Orange
        color_arm = [46 / 255.0, 117 / 255.0, 182 / 255.0]  # Blue

        def create_o3d_box(extents, translation, color):
            """Helper to create an Open3D box, center it, and color it."""
            # Open3D creates a box from (0,0,0) to (width, height, depth)
            box = o3d.geometry.TriangleMesh.create_box(
                width=extents[0], height=extents[1], depth=extents[2]
            )
            # 1. Translate so the center of the box is at (0,0,0)
            box.translate([-extents[0] / 2.0, -extents[1] / 2.0, -extents[2] / 2.0])
            # 2. Translate to the actual target position
            box.translate(translation)
            # 3. Compute normals for lighting and paint it
            box.compute_vertex_normals()
            box.paint_uniform_color(color)
            return box

        # --- 1. Fingers (Purple) ---
        finger_len = c.b_pg + c.c_pg
        finger_extents = [c.e_pg, c.a_pg, finger_len]

        # Left Finger (+Y position)
        finger_l = create_o3d_box(
            finger_extents,
            [0, (c.f_pg / 2) + (c.a_pg / 2), finger_len / 2],
            color_finger,
        )
        meshes.append(finger_l)

        # Right Finger (-Y position)
        finger_r = create_o3d_box(
            finger_extents,
            [0, -(c.f_pg / 2) - (c.a_pg / 2), finger_len / 2],
            color_finger,
        )
        meshes.append(finger_r)

        current_z = finger_len

        # --- 2. Gripper Base Bottom (Orange) ---
        base_bottom = create_o3d_box(
            [c.l_pg, c.h_pg, c.t_pg], [0, 0, current_z + (c.t_pg / 2)], color_base_bot
        )
        meshes.append(base_bottom)
        current_z += c.t_pg

        # --- 3. Gripper Base Top (Darker Orange) ---
        base_top = create_o3d_box(
            [c.o_pg, c.q_pg, c.u_pg], [0, 0, current_z + (c.u_pg / 2)], color_base_top
        )
        meshes.append(base_top)
        current_z += c.u_pg

        # --- 4. Robot Arm Link (Blue) ---
        robot_link = create_o3d_box(
            [c.rb, c.ra, c.rc], [0, 0, current_z + (c.rc / 2)], color_arm
        )
        meshes.append(robot_link)

        # Merge all Open3D parts into a single mesh
        merged_o3d = gu.merge_meshes(meshes)

        return GenericGeometry(geometry=merged_o3d)

    def generate_safety_collision_mesh(
        self,
        contact_point: np.ndarray,
        surface_normal: np.ndarray,
        approach_distance: float,
    ) -> trimesh.Trimesh:
        """
        Generates the 3D collision volume at a specific grasp pose.
        Requires returning a Trimesh object for the collision_manager in the sampler.
        """
        mesh = self.collision_geometry.as_trimesh().copy()

        # Align gripper Z-axis with the surface normal (pointing OUT)
        R = gu.get_rotation_matrix_between_vectors(np.array([0, 0, 1]), surface_normal)
        T = gu.create_pose_matrix(R, contact_point)
        mesh.apply_transform(T)

        return mesh
