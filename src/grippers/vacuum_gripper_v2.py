import copy
import os
import yaml
import numpy as np
import open3d as o3d
import trimesh
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.grippers.base_gripper import BaseGripper
from src.generic_geometry import GenericGeometry
import src.utils.geometry_utils as gu
from src.grippers.pads import (
    BaseSuctionPad,
    CircularPad,
    CircularPadZones,
    RectangularPad,
)

# Standard Colors
CUP_COLOR = [0.0, 0.8, 0.0]  # Green
BODY_COLOR = [0.5, 0.5, 0.5]  # Grey


@dataclass
class VacuumGripperConfig:
    """
    Configuration data structure.
    Holds both physical components (pads) and visual assets.
    """

    pads: List[BaseSuctionPad] = field(default_factory=list)
    pad_height: float = 0.04
    visual_mesh_path: Optional[str] = None
    collision_mesh_path: Optional[str] = None
    body_radius: float = 0.075
    body_length: float = 0.088
    standoff_distance: float = 0.05
    collision_margin: float = 0.05
    name: str = "UnknownVacuum"
    grasp_strategy: str = "projection"


class VacuumGripper(BaseGripper):
    """
    Concrete implementation of a Vacuum Gripper.
    Refactored to support Multi-Pad configurations and External Assets.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # 1. Load Configuration (Decomposed into sub-methods)
        self.config: VacuumGripperConfig = self.load_config()

        # 2. Build or Load 3D Model
        self.collision_geometry = self.generate_collision_mesh()

        self._log_init()

    def _log_init(self):
        print(f"🔧 Vacuum Gripper '{self.config.name}' initialized.")
        print(f"   - Active Pads: {len(self.config.pads)}")
        if self.config.visual_mesh_path:
            print(
                f"   - External Asset: {os.path.basename(self.config.visual_mesh_path)}"
            )

    # =========================================================================
    # 1. Configuration Loading Logic
    # =========================================================================
    def load_config(self) -> VacuumGripperConfig:
        """
        Parses the YAML file into a typed configuration object.
        """
        raw_data = self._read_yaml(self.config_path)
        gripper_data = raw_data.get("gripper", {})

        # 1. Parse Basic Config first (to get the global pad_height)
        basic_config = self._parse_basic_attributes(gripper_data)

        # 2. Parse Pads, injecting the global pad_height as default length
        basic_config.pads = self._parse_pads_list(
            gripper_data,
            default_length=basic_config.pad_height,  # <--- Passando o valor global
        )

        return basic_config

    def _read_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _parse_basic_attributes(self, data: Dict[str, Any]) -> VacuumGripperConfig:
        """Extracts simple scalar values from the config dict."""
        body = data.get("body", {})
        safety = data.get("safety", {})

        return VacuumGripperConfig(
            name=data.get("name", "UnknownVacuum"),
            visual_mesh_path=data.get("visual_mesh_path"),
            collision_mesh_path=data.get("collision_mesh_path"),
            body_radius=float(body.get("radius", 0.075)),
            body_length=float(body.get("length", 0.088)),
            standoff_distance=float(safety.get("standoff_distance", 0.05)),
            collision_margin=float(safety.get("collision_margin", 0.05)),
            grasp_strategy=data.get("grasp_strategy", "single"),
            pad_height=float(data.get("pad_height", 0.04)),
        )

    def _parse_pads_list(
        self, data: Dict[str, Any], default_length: float
    ) -> List[BaseSuctionPad]:
        """
        Parses the 'pads' list and enforces the global length.
        """
        pads_data = data.get("pads", [])

        if pads_data:
            pads_obj = []
            for p_data in pads_data:
                # INJECTION: Force or Default the length to match global config
                if "length" not in p_data:
                    p_data["length"] = default_length

                pads_obj.append(BaseSuctionPad.from_config(p_data))
            return pads_obj
        else:
            # Legacy Mode
            return [
                self._create_legacy_pad(data.get("suction_cup", {}), default_length)
            ]

    def _create_legacy_pad(
        self, cup_data: Dict[str, Any], default_length: float
    ) -> BaseSuctionPad:
        """Creates a default center pad for old YAML files."""
        return CircularPad(
            name="center_cup",
            offset=np.array([0, 0, 0]),
            radius=float(cup_data.get("radius", 0.02)),
            length=float(
                cup_data.get("length", default_length)
            ),  # Usa global se faltar
            zones=CircularPadZones(num_radial_sections=2, num_angular_sections=4),
        )

    # =========================================================================
    # 2. Geometry Generation (Visual / Collision Body)
    # =========================================================================
    def generate_collision_mesh(self) -> GenericGeometry:
        """
        Generates the static geometry of the gripper and applies the Pad Height offset.

        CRITICAL:
        The geometry is shifted along the Z-axis by 'pad_height'.
        This ensures that the TCP (0,0,0) corresponds to the TIP of the suction cups,
        not the mounting flange or the base of the body.
        """
        geometry_wrapper = None

        # 1. Try loading an external mesh file first
        if self.config.visual_mesh_path:
            # gu.load_mesh_file returns a Trimesh or Open3D object, or None
            mesh = gu.load_mesh_file(self.config.visual_mesh_path)
            if mesh is not None:
                geometry_wrapper = GenericGeometry(geometry=mesh)

        # 2. Fallback to procedural generation (Primitive shapes) if no file is found
        if geometry_wrapper is None:
            geometry_wrapper = self._build_procedural_geometry()

        # 3. Apply Z-Axis Offset (Pad Height Adjustment)
        # We need to move the entire gripper body UP (positive Z) so that the
        # origin (0,0,0) aligns with the suction cup tips.
        # We use the 'transform' method from GenericGeometry which handles
        # the underlying library (Open3D/Trimesh) abstraction automatically.

        offset_vector = np.array([0.0, 0.0, self.config.pad_height])

        try:
            # Passes the translation vector. The wrapper creates the matrix internally.
            geometry_wrapper.transform(translation=offset_vector)
        except Exception as e:
            print(f"[Warning] Could not apply offset to collision mesh: {e}")

        return geometry_wrapper

    # def generate_collision_mesh(self) -> GenericGeometry:
    #     """
    #     Generates the static geometry of the gripper (at origin).
    #     Priority: 1. External Mesh File -> 2. Procedural Generation.
    #     """
    #     # Try loading external file first
    #     if self.config.visual_mesh_path:
    #         mesh = gu.load_mesh_file(self.config.visual_mesh_path)
    #         if mesh:
    #             return GenericGeometry(geometry=mesh)
    #
    #     # Fallback to procedural generation (The "Lego" construction)
    #     return self._build_procedural_geometry()

    def _build_procedural_geometry(self) -> GenericGeometry:
        meshes = []
        meshes.extend(self._generate_pad_meshes())
        meshes.append(self._generate_body_mesh())

        # Merge all parts into one geometry
        full_mesh = gu.merge_meshes(meshes)
        return GenericGeometry(geometry=full_mesh)

    def _generate_pad_meshes(self) -> List[o3d.geometry.TriangleMesh]:
        pad_meshes = []
        for pad in self.config.pads:
            # Polymorphism: Ask the pad for its own geometry
            trimesh_geom = pad.get_collision_mesh()

            # Convert Trimesh -> Open3D for visualization
            o3d_geom = o3d.geometry.TriangleMesh()
            o3d_geom.vertices = o3d.utility.Vector3dVector(trimesh_geom.vertices)
            o3d_geom.triangles = o3d.utility.Vector3iVector(trimesh_geom.faces)
            o3d_geom.compute_vertex_normals()
            o3d_geom.paint_uniform_color(CUP_COLOR)

            pad_meshes.append(o3d_geom)
        return pad_meshes

    def _generate_body_mesh(self) -> o3d.geometry.TriangleMesh:
        """Generates the main cylinder body of the gripper."""
        return gu.create_cylinder(
            radius=self.config.body_radius,
            length=self.config.body_length,
            start_point=np.array([0, 0, 0.0]),  # Offset slightly above pads
            direction_vector=np.array([0, 0, 1]),
            color=BODY_COLOR,
            backend="open3d",
        )

    # =========================================================================
    # 3. Safety Volume (Collision Checking Logic)
    # =========================================================================
    def generate_safety_collision_mesh(
        self,
        contact_point: np.ndarray,
        surface_normal: np.ndarray,
        approach_distance: float,
    ) -> trimesh.Trimesh:
        """
        Generates the collision volume for path checking.

        Refined Logic:
        1. Pads: STATIC geometry (inflated). Not extruded, to allow close contact.
        2. Body: SWEPT volume (tube). Extruded backwards to check approach path.
        """
        safety_margin = 1.0 + self.config.collision_margin
        parts = []

        # --- A. Setup Transformation Matrix (Align Z with Normal) ---
        # The safety mesh is generated at the TCP (contact_point) oriented along the normal.
        R_align = gu.get_rotation_matrix_between_vectors(
            np.array([0, 0, 1]), surface_normal
        )
        T_align = np.eye(4)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = contact_point

        # --- B. Generate STATIC Safety Mesh for Pads ---
        for pad in self.config.pads:
            # 1. Get the mesh from the pad itself (Polymorphism: Rect or Circle)
            # This mesh comes defined in local coordinates relative to the TCP (including offset)
            pad_mesh = pad.get_safety_mesh(margin_scale=safety_margin)

            # 2. Transform it to the World Pose (Contact Point + Orientation)
            pad_mesh.apply_transform(T_align)
            pad_mesh.visual.face_colors = [int(c * 255) for c in CUP_COLOR] + [
                200
            ]  # RGBA with transparency

            parts.append(pad_mesh)

        # --- C. Generate SWEPT Safety Volume for Body ---
        # The body needs to clear the path BEHIND the pads.
        parts.append(
            self._create_body_volume(
                contact_point, surface_normal, approach_distance, safety_margin
            )
        )

        return gu.merge_meshes(parts)

    def _create_body_volume(
        self, tcp: np.ndarray, normal: np.ndarray, dist: float, margin: float
    ) -> trimesh.Trimesh:
        """Creates the swept volume (tube) for the main body."""

        return gu.create_cylinder(
            radius=self.config.body_radius * margin,
            length=dist,
            start_point=tcp,
            direction_vector=normal,
            color=BODY_COLOR,
            backend="trimesh",
        )

    def _calculate_pad_world_pos(
        self, pad: BaseSuctionPad, tcp: np.ndarray, normal: np.ndarray
    ) -> np.ndarray:
        """
        Projects the local pad offset into the world frame, aligned with the surface normal.
        """
        # Calculate rotation matrix that aligns Z-axis with the surface normal
        R = gu.get_rotation_matrix_between_vectors(np.array([0, 0, 1]), normal)

        # Rotate the offset vector and add to TCP
        world_offset = R @ pad.offset
        return tcp + world_offset
