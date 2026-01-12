import abc
import numpy as np
import trimesh
from dataclasses import dataclass


class BaseSuctionPad(abc.ABC):
    """
    Abstract Base Class for a generic Suction Pad.
    Follows Open/Closed Principle: Extend this class to add new shapes.
    """

    def __init__(self, name: str, offset: np.ndarray):
        """
        Args:
            name: ID for debugging (e.g., "left_cup")
            offset: [x, y, z] position relative to the Gripper TCP
        """
        self.name = name
        self.offset = np.array(offset, dtype=np.float64)

    @abc.abstractmethod
    def is_point_inside(self, local_points_xy: np.ndarray) -> np.ndarray:
        """
        Determines which points fall inside the pad's sealing area.

        Args:
            local_points_xy: Nx2 array of points relative to the pad center.

        Returns:
            Boolean mask (N,) where True = Inside.
        """
        pass

    @abc.abstractmethod
    def get_collision_mesh(self) -> trimesh.Trimesh:
        """
        Returns the collision geometry (Safety Volume) for this specific pad.
        """
        pass


@dataclass
class CircularPad(BaseSuctionPad):
    """
    Standard circular suction cup.
    """

    radius: float

    def __init__(self, name: str, offset: np.ndarray, radius: float):
        super().__init__(name, offset)
        self.radius = radius

    def is_point_inside(self, local_points_xy: np.ndarray) -> np.ndarray:
        # Math: x^2 + y^2 <= r^2
        dists = np.linalg.norm(local_points_xy, axis=1)
        return dists <= self.radius

    def get_collision_mesh(self) -> trimesh.Trimesh:
        # Create cylinder
        # Height 0.05 is an arbitrary safety length for the tube
        mesh = trimesh.creation.cylinder(radius=self.radius, height=0.05)

        # Move to offset position
        mesh.apply_translation(self.offset)
        return mesh


@dataclass
class RectangularPad(BaseSuctionPad):
    """
    Rectangular foam pad (common in vacuum area grippers).
    """

    width: float  # Dimension in X
    height: float  # Dimension in Y

    def __init__(self, name: str, offset: np.ndarray, width: float, height: float):
        super().__init__(name, offset)
        self.width = width
        self.height = height

    def is_point_inside(self, local_points_xy: np.ndarray) -> np.ndarray:
        # Math: |x| <= w/2 AND |y| <= h/2
        x = np.abs(local_points_xy[:, 0])
        y = np.abs(local_points_xy[:, 1])
        return (x <= self.width / 2.0) & (y <= self.height / 2.0)

    def get_collision_mesh(self) -> trimesh.Trimesh:
        # Create box
        mesh = trimesh.creation.box(extents=[self.width, self.height, 0.05])

        # Move to offset position
        mesh.apply_translation(self.offset)
        return mesh


# Future extensions:
# class OvalPad(BaseSuctionPad): ...
# class DonutPad(BaseSuctionPad): ...
