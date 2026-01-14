import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional
import open3d as o3d
import trimesh
import copy

# Adjust path to find core modules if necessary, or use relative imports
# Assuming structure: project/grippers/ and project/core/
from src.generic_geometry import GenericGeometry


class BaseGripper(ABC):
    """
    Abstract Base Class that defines the mandatory interface for any gripper type
    (e.g., Vacuum, Parallel, Magnetic, Soft).

    This ensures that the Estimator class can interact with any gripper
    polymorphically, without knowing the specific implementation details.
    """

    def __init__(self, config_path: str):
        """
        Initialize the gripper.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path

        # Every gripper must have a geometric representation for collision checking
        self.collision_geometry: Optional[GenericGeometry] = None

        # Configuration data object (structure depends on the specific gripper)
        self.config: Any = None

    @abstractmethod
    def load_config(self) -> Any:
        """
        Must implement logic to load dimensions and parameters from the configuration file.

        Returns:
            A dataclass or dictionary with the specific gripper parameters.
        """
        pass

    @abstractmethod
    def generate_collision_mesh(self) -> GenericGeometry:
        """
        Must generate the 3D geometry of the gripper based on loaded parameters.

        CRITICAL: The geometry must be generated such that the
        TCP (Tool Center Point) is located exactly at (0, 0, 0).

        Returns:
            GenericGeometry: The 3D model of the gripper.
        """
        pass

    @abstractmethod
    def generate_safety_collision_mesh(
        self,
        contact_point: np.ndarray,
        surface_normal: np.ndarray,
        approach_distance: float,
    ) -> trimesh.Trimesh:
        """
        Generates the swept volume (safety tube) for collision checking.
        """
        pass

    def visualize(self):
        assert self.collision_geometry is not None, "Collision geometry not generated."
        self.collision_geometry.visualize()
