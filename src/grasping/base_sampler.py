from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing import Any, List, Generic, TypeVar, Optional, TypedDict
from typing_extensions import Dict

import numpy as np
import open3d as o3d
import trimesh

from src.grippers.base_gripper import BaseGripper

# --- 1. Define Generic Type Variables ---
# TGripper must be a subclass of BaseGripper
TGripper = TypeVar("TGripper", bound=BaseGripper)
# TConfig can be any type (usually a dataclass), we don't strictly bind it here
# to allow flexibility, but you could bind it to a BaseSamplerConfig if you have one.
TConfig = TypeVar("TConfig")
TScoreDetails = TypeVar("TScoreDetails")


@dataclass
class GraspCandidate(Generic[TScoreDetails]):
    """
    Data structure representing a potential grasp solution.
    """

    transform: np.ndarray  # 4x4 Homogeneous Transformation Matrix (Pose)
    score: float  # 0.0 to 1.0 (Quality/Probability of Success)
    contact_point: np.ndarray  # XYZ coordinates on the object surface
    approach_vector: np.ndarray  # Normalized vector representing approach direction
    score_details: TScoreDetails


class BaseGraspSampler(Generic[TGripper, TConfig, TScoreDetails], metaclass=ABCMeta):
    """
    Abstract Base Class for Grasp Sampling algorithms using Generics.

    By inheriting from Generic[TGripper, TConfig], child classes automatically
    know the specific types of self.gripper and self.config.
    """

    def __init__(self, gripper: TGripper, config: TConfig):
        """
        Args:
            gripper: The specific gripper instance (e.g., VacuumGripper).
            config: The specific configuration (e.g., VacuumSamplerConfig).
        """
        self.gripper: TGripper = gripper
        self.config: TConfig = config
        self.candidates: List[GraspCandidate[TScoreDetails]] = []

        # Trimesh Collision Manager (The "Judge")
        self.collision_manager = trimesh.collision.CollisionManager()

    def update_environment(self, obstacles: List[trimesh.Trimesh]):
        """
        Updates the internal collision world with static obstacles.
        """
        self.collision_manager = trimesh.collision.CollisionManager()
        for i, mesh in enumerate(obstacles):
            self.collision_manager.add_object(f"env_obstacle_{i}", mesh)

        print(
            f"[BaseGraspSampler] Environment updated with {len(obstacles)} obstacles."
        )

    def check_collision(self, safety_mesh: trimesh.Trimesh) -> bool:
        """
        Generic collision check.
        """
        col = self.collision_manager.in_collision_single(safety_mesh)
        assert isinstance(col, bool)
        return col

    @abstractmethod
    def sample_grasps(self, pcd: o3d.geometry.PointCloud) -> List[GraspCandidate]:
        """
        Core logic to be implemented by child classes.
        """
        pass

    def clear_candidates(self):
        """
        Resets the internal list of grasp candidates.
        """
        self.candidates = []
        self.valid_candidates = []
