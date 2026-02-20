import yaml
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Any, Optional

from src.grippers.base_gripper import BaseGripper
from src.generic_geometry import GenericGeometry
import src.utils.geometry_utils as gu


@dataclass
class ParallelGripperConfig:
    """Dataclass mapping the Franka.yaml parameters."""

    # Finger Widths
    a_pg: float  # Finger width
    w_pg: float  # Internal Safespace Finger width
    v_pg: float  # External Safespace Finger width

    # Gripper Opening
    f_pg: float  # Distance gripper open
    g_pg: float  # Distance gripper close

    # Gripper Base Widths
    h_pg: float  # Gripper base bottom width
    k_pg: float  # Safespace Gripper base bottom width
    q_pg: float  # Gripper base top width
    r_pg: float  # Safespace Gripper base top width

    # Lengths
    b_pg: float  # Gripper area length end
    c_pg: float  # Gripper area to (Safety space) length end
    d_pg: float  # Safespace Gripper length
    x_pg: float  # Safespace Gripper end to rubber
    t_pg: float  # Gripper base bottom length
    u_pg: float  # Gripper base top length

    # Depths
    e_pg: float  # Finger depth
    i_pg: float  # Safespace finger depth
    z_pg: float  # Gripper area depth
    l_pg: float  # Gripper base bottom depth
    m_pg: float  # Safespace gripper base bottom depth
    o_pg: float  # Gripper base top depth
    p_pg: float  # Safespace gripper base top depth

    # Robot Arm constraints
    ra: float  # width of last robot arm limb
    rb: float  # depth of last robot arm limb
    rc: float  # length of last robot arm limb
    re: float  # robot arm diameter clearance
    rf: float  # robot arm length clearance
    rj: float  # repeatability of robot arm


class ParallelGripper(BaseGripper):
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

        # Opcional: Se o YAML estiver em mm, descomente a conversão
        # MM_TO_M = 0.001
        # data = {k: (v * MM_TO_M if isinstance(v, (int, float)) else v) for k, v in data.items()}

        return ParallelGripperConfig(**data)

    def generate_collision_mesh(self) -> GenericGeometry:
        """
        Gera uma representação 3D básica do gripper paralelo para visualização.
        O TCP (0,0,0) fica no centro entre os dedos.
        """
        c = self.config
        meshes = []

        # Dedos (Esquerdo e Direito) - Simplificados como caixas
        finger_extents = [c.e_pg, c.a_pg, c.b_pg + c.c_pg]

        # Dedo Esquerdo
        finger_l = trimesh.creation.box(extents=finger_extents)
        finger_l.apply_translation(
            [0, (c.f_pg / 2) + (c.a_pg / 2), -(c.b_pg + c.c_pg) / 2]
        )
        meshes.append(finger_l)

        # Dedo Direito
        finger_r = trimesh.creation.box(extents=finger_extents)
        finger_r.apply_translation(
            [0, -(c.f_pg / 2) - (c.a_pg / 2), -(c.b_pg + c.c_pg) / 2]
        )
        meshes.append(finger_r)

        # Base do Gripper
        base_extents = [c.l_pg, c.h_pg, c.t_pg]
        base = trimesh.creation.box(extents=base_extents)
        base.apply_translation([0, 0, -(c.b_pg + c.c_pg + (c.t_pg / 2))])
        meshes.append(base)

        merged = gu.merge_meshes(meshes)
        merged.visual.face_colors = [128, 128, 128, 255]  # Cinza

        return GenericGeometry(geometry=merged)

    def generate_safety_collision_mesh(
        self,
        contact_point: np.ndarray,
        surface_normal: np.ndarray,
        approach_distance: float,
    ) -> trimesh.Trimesh:
        """
        Gera o volume de colisão 3D (Opcional se o Sampler usar a lógica Shapely 2D).
        Podemos apenas retornar o mesh dilatado por enquanto.
        """
        mesh = self.collision_geometry.as_trimesh().copy()
        # Aplica a pose básica
        R = gu.get_rotation_matrix_between_vectors(np.array([0, 0, 1]), surface_normal)
        T = gu.create_pose_matrix(R, contact_point)
        mesh.apply_transform(T)
        return mesh
