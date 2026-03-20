import open3d as o3d
import numpy as np
import trimesh
import yaml
from typing import Literal, Optional, Union
from dataclasses import dataclass
from src.grippers.base_gripper import BaseGripper
from src.generic_geometry import GenericGeometry
import src.utils.geometry_utils as gu

CUP_MESH_COLOR = [0.0, 0.8, 0.0]  # Green
BODY_MESH_COLOR = [0.5, 0.5, 0.5]  # Grey


@dataclass
class VacuumGripperConfig:
    """
    Data structure holding the physical dimensions of the vacuum gripper.
    """

    cup_radius: float  # Radius of the suction cup (m)
    cup_height: float  # Height of the suction cup (m)
    body_radius: float  # Radius of the main housing (m)
    body_length: float  # Length of the main housing (m)
    standoff_distance: float
    collision_margin: float  # Safety margin for collision checking (fraction)
    name: str = "Unknown"


class VacuumGripper(BaseGripper):
    """
    Concrete implementation of a Vacuum Gripper (e.g., Schmalz ECG).
    Models the geometry as two stacked cylinders:
    1. The Suction Cup (Green)
    2. The Main Body (Grey)
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # 1. Load Dimensions
        self.config = self.load_config()

        # 2. Build 3D Model
        self.collision_geometry = self.generate_collision_mesh()

        print(f"🔧 Vacuum Gripper '{self.config.name}' initialized.")
        print(
            f"   - Body Dimensions: R={self.config.body_radius * 1000:.1f}mm, L={self.config.body_length * 1000:.1f}mm"
        )

    def load_config(self) -> VacuumGripperConfig:
        """
        Parses the YAML file based on the structure defined in config/franka_vacuum.yaml.
        """
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        g_data = data.get("gripper", {})
        cup_data = g_data.get("suction_cup", {})
        body_data = g_data.get("body", {})
        safety_data = g_data.get("safety", {})

        return VacuumGripperConfig(
            cup_radius=float(cup_data.get("radius", 0.02)),
            cup_height=float(cup_data.get("height", 0.03)),
            body_radius=float(body_data.get("radius", 0.075)),
            body_length=float(body_data.get("length", 0.088)),
            standoff_distance=float(safety_data.get("standoff_distance", 0.05)),
            collision_margin=float(safety_data.get("collision_margin", 0.05)),
            name=g_data.get("name", "UnknownVacuum"),
        )

    def generate_collision_mesh(self) -> GenericGeometry:
        """
        Generates the VISUAL geometry (Open3D).
        """
        # Backend padrão é 'open3d'
        cup_mesh: o3d.geometry.TriangleMesh = self._make_cup_mesh(backend="open3d")
        body_start_pos = np.array([0, 0, self.config.cup_height])
        body_mesh: o3d.geometry.TriangleMesh = self._make_body_mesh(
            start_point=body_start_pos, backend="open3d"
        )

        # USA O NOVO MERGE (funciona com lista de Open3D)
        full_mesh: o3d.geometry.TriangleMesh = gu.merge_meshes([cup_mesh, body_mesh])

        # Importante: recalcular normais após o merge para a luz bater certo
        full_mesh.compute_vertex_normals()

        full_mesh = GenericGeometry(geometry=full_mesh)
        return full_mesh

    def generate_safety_collision_mesh(
        self,
        contact_point: np.ndarray,
        surface_normal: np.ndarray,
        approach_distance: float,
    ) -> trimesh.Trimesh:
        """
        Generates the COLLISION geometry (Trimesh).
        """
        factor = 1.0 + self.config.collision_margin
        r_cup_safe = self.config.cup_radius * factor
        r_body_safe = self.config.body_radius * factor

        # 1. Suction Cup
        mesh_cup = gu.create_cylinder(
            start_point=contact_point,
            direction_vector=surface_normal,
            length=self.config.cup_height,
            radius=r_cup_safe,
            backend="trimesh",  # Forces Trimesh backend
            color=CUP_MESH_COLOR,  # Blue RGBA for trimesh
        )

        # 2. Body
        start_body = contact_point + (surface_normal * self.config.cup_height)
        mesh_body = gu.create_cylinder(
            start_point=start_body,
            direction_vector=surface_normal,
            length=approach_distance,
            radius=r_body_safe,
            backend="trimesh",  # Força backend Trimesh
            color=BODY_MESH_COLOR,
        )

        # USA O NOVO MERGE (funciona com lista de Trimesh)
        full_mesh = gu.merge_meshes([mesh_cup, mesh_body])

        return full_mesh

    def _make_cup_mesh(
        self,
        safe_mode: bool = False,
        start_point: np.ndarray = np.array([0, 0, 0]),
        direction_vector: np.ndarray = np.array([0, 0, 1.0]),
        backend: Literal["open3d", "trimesh"] = "open3d",
    ) -> Union[o3d.geometry.TriangleMesh, trimesh.Trimesh]:
        """
        Returns the mesh of the suction cup only.
        """
        radius = self.config.cup_radius
        radius = radius * (1.0 + self.config.collision_margin) if safe_mode else radius
        cup_mesh = gu.create_cylinder(
            length=self.config.cup_height,
            radius=radius,
            start_point=start_point,
            direction_vector=direction_vector,
            color=CUP_MESH_COLOR,
            backend=backend,
        )
        return cup_mesh

    def _make_body_mesh(
        self,
        safe_mode: bool = False,
        start_point: np.ndarray = np.array([0, 0, 0]),
        direction_vector: np.ndarray = np.array([0, 0, 1.0]),
        backend: Literal["open3d", "trimesh"] = "open3d",
        extension: Optional[float] = None,
    ) -> Union[o3d.geometry.TriangleMesh, trimesh.Trimesh]:
        """
        Returns the mesh of the main body only.
        """
        radius = self.config.body_radius
        radius = radius * (1.0 + self.config.collision_margin) if safe_mode else radius
        height = extension if extension is not None else self.config.body_length
        body_mesh = gu.create_cylinder(
            length=height,
            radius=radius,
            start_point=start_point,
            direction_vector=direction_vector,
            color=BODY_MESH_COLOR,
            backend=backend,
        )
        return body_mesh

    def _assemble_full_mesh(
        self,
        cup_mesh: o3d.geometry.TriangleMesh,
        body_mesh: o3d.geometry.TriangleMesh,
        body_length: Optional[float] = None,
    ) -> o3d.geometry.TriangleMesh:
        """
        Assembles the full gripper mesh from cup and body meshes.
        """
        body_length = (
            body_length if body_length is not None else self.config.body_length
        )
        assert body_length is not None, "Body mesh cannot be None."
        body_center_z = self.config.cup_height + (body_length / 2.0)
        body_mesh.translate([0, 0, body_center_z])
        full_mesh = cup_mesh + body_mesh
        full_mesh.compute_vertex_normals()
        return full_mesh

    def visualize(self):
        assert self.collision_geometry is not None, "Collision geometry not generated."
        self.collision_geometry.visualize()
