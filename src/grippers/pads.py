from __future__ import annotations
import abc
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Dict, Any, Type


class BaseSuctionPad(abc.ABC):
    """
    Abstract Base Class for a generic Suction Pad.
    Follows Open/Closed Principle: New shapes can be added by extending this class,
    without modifying the main Gripper code.
    """

    def __init__(self, name: str, offset: np.ndarray, lenght: float = 0.0):
        """
        Args:
            name: Identifier for debugging (e.g., "left_cup").
            offset: [x, y, z] position relative to the Gripper TCP.
        """
        self.name = name
        self.offset = np.array(offset, dtype=np.float64)
        self.length = lenght

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
        Returns the specific collision geometry for this pad.
        """
        pass

    @property
    @abc.abstractmethod
    def safety_radius(self) -> float:
        """
        Returns the effective radius for safety volume calculations.
        Allows the Gripper to calculate approach tubes without knowing the exact shape.
        """
        pass

    @abc.abstractmethod
    def get_safety_mesh(self, margin_scale: float) -> trimesh.Trimesh:
        """
        Returns the collision mesh inflated by the safety margin.
        Used for static collision checking near the object surface.
        """
        pass

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "BaseSuctionPad":
        """
        Factory Method: Creates the correct Pad instance based on the 'type' field.
        This moves the conditional logic out of the Gripper class.
        """
        # Create a copy to avoid modifying the original dictionary
        cfg = config.copy()
        pad_type = cfg.pop("type", "circle")

        # Ensure offset is a numpy array
        if "offset" in cfg:
            cfg["offset"] = np.array(cfg["offset"])

        global REGISTRY
        if pad_type not in REGISTRY:
            raise ValueError(
                f"Unknown pad type: {pad_type}. Available: {list(REGISTRY.keys())}"
            )

        pad_class = REGISTRY[pad_type]
        # Unpack remaining config items as arguments for the specific class constructor
        return pad_class(**cfg)


@dataclass
class CircularPad(BaseSuctionPad):
    """
    Standard circular suction cup.
    """

    radius: float
    length: float

    def __init__(self, name: str, offset: np.ndarray, radius: float, length: float):
        super().__init__(name, offset)
        self.radius = radius
        self.length = length

    def is_point_inside(self, local_points_xy: np.ndarray) -> np.ndarray:
        # Math: x^2 + y^2 <= r^2
        dists = np.linalg.norm(local_points_xy, axis=1)
        return dists <= self.radius

    def get_collision_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.cylinder(radius=self.radius, height=self.length)
        mesh.apply_translation(self.offset)
        mesh.apply_translation([0, 0, -self.length / 2.0])  # Center at base
        return mesh

    @property
    def safety_radius(self) -> float:
        return self.radius

    def get_safety_mesh(self, margin_scale: float) -> trimesh.Trimesh:
        # Apenas infla o raio. A altura (length) mantemos original para não
        # criar colisão fantasma com o objeto.
        safe_radius = self.radius * margin_scale

        mesh = trimesh.creation.cylinder(radius=safe_radius, height=self.length)

        # Centraliza e aplica offset local (mesma lógica do collision normal)
        # Cilindro nasce no centro (0,0,0). Movemos para base (-length/2) e aplicamos offset.
        mesh.apply_translation([0, 0, -self.length / 2.0])
        mesh.apply_translation(self.offset)
        return mesh


@dataclass
class RectangularPad(BaseSuctionPad):
    """
    Rectangular foam pad (common in area grippers).
    """

    width: float
    height: float
    length: float

    def __init__(
        self, name: str, offset: np.ndarray, width: float, height: float, length: float
    ):
        super().__init__(name, offset)
        self.width = width
        self.height = height
        self.length = length

    def is_point_inside(self, local_points_xy: np.ndarray) -> np.ndarray:
        # Math: |x| <= w/2 AND |y| <= h/2
        x = np.abs(local_points_xy[:, 0])
        y = np.abs(local_points_xy[:, 1])
        return (x <= self.width / 2.0) & (y <= self.height / 2.0)

    def get_collision_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(extents=[self.width, self.height, self.length])
        mesh.apply_translation(self.offset)
        mesh.apply_translation([0, 0, -self.length / 2.0])
        return mesh

    def get_safety_mesh(self, margin_scale: float) -> trimesh.Trimesh:
        safe_w = self.width * margin_scale
        safe_h = self.height * margin_scale

        mesh = trimesh.creation.box(extents=[safe_w, safe_h, self.length])

        # Ajusta posição
        mesh.apply_translation([0, 0, -self.length / 2.0])
        mesh.apply_translation(self.offset)
        return mesh

    @property
    def safety_radius(self) -> float:
        # Use the largest dimension to ensure a conservative safety volume
        return max(self.width, self.height)


REGISTRY: Dict[str, Type[BaseSuctionPad]] = {
    "circle": CircularPad,
    "rect": RectangularPad,
}
