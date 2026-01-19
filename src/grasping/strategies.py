import abc
import numpy as np
import open3d as o3d
from typing import List, Dict, Type

# Import BaseSuctionPad for type hinting
from src.grippers.pads import BaseSuctionPad


class GraspContactStrategy(abc.ABC):
    """
    Abstract Base Class for Grasp Contact Strategies.

    Responsibility:
    Receives a candidate Pose (TCP + Rotation) and determines WHERE
    each suction pad would touch the object's surface.

    This decouples the geometric logic (Projection, Physical Adjustment, etc.)
    from the physical evaluation logic (GSS / Vacuum Score).
    """

    @abc.abstractmethod
    def resolve_contacts(
        self,
        tcp_pose: np.ndarray,
        pads: List[BaseSuctionPad],
        pcd_tree: o3d.geometry.KDTreeFlann,
        all_points: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Calculates the actual contact points on the surface for each pad.

        Args:
            tcp_pose: 4x4 homogeneous transformation matrix of the TCP.
            pads: List of Pad objects (containing their offsets).
            pcd_tree: KDTree of the point cloud for fast spatial searches.
            all_points: (N, 3) array containing the coordinates of the cloud points.

        Returns:
            List[np.ndarray]: A list of (x, y, z) points on the surface where each pad
                              would theoretically make contact. The order corresponds
                              to the input 'pads' list.
        """
        pass


class SinglePadStrategy(GraspContactStrategy):
    """
    Strategy for Single Suction Cup Grippers.

    Logic:
    The contact point is simply the TCP translation itself.
    There is no need for complex projection because the TCP is already
    positioned at the test point on the surface.
    """

    def resolve_contacts(
        self,
        tcp_pose: np.ndarray,
        pads: List[BaseSuctionPad],
        pcd_tree: o3d.geometry.KDTreeFlann,
        all_points: np.ndarray,
    ) -> List[np.ndarray]:
        # The contact is the TCP origin itself (column 3 of the transformation matrix)
        tcp_origin = tcp_pose[:3, 3]

        # Returns a list containing the single contact point
        return [tcp_origin]


class MultiPadProjectionStrategy(GraspContactStrategy):
    """
    Projection Strategy (Justus' Idea 2).
    Recommended for rigid Multi-Pad Grippers.

    Logic:
    1. Keeps the gripper 'frozen' at the candidate pose.
    2. Calculates the theoretical rigid position of each cup in the world
       (using the fixed offset).
    3. Projects this position onto the object's surface (Nearest Neighbor).

    This allows measuring the 'gap' (error) of each cup individually
    without physically moving the robot simulation.
    """

    def resolve_contacts(
        self,
        tcp_pose: np.ndarray,
        pads: List[BaseSuctionPad],
        pcd_tree: o3d.geometry.KDTreeFlann,
        all_points: np.ndarray,
    ) -> List[np.ndarray]:
        contact_points = []

        # Extract Rotation (R) and Translation (t) from the TCP Pose
        R_tcp = tcp_pose[:3, :3]
        t_tcp = tcp_pose[:3, 3]

        for pad in pads:
            # 1. Rigid World Position (Where the cup is 'floating' in space)
            # Formula: P_world = TCP_pos + (R_tcp @ Pad_offset)
            pad_world_pos = t_tcp + (R_tcp @ pad.offset)

            # 2. Projection onto Surface (Nearest Neighbor Search)
            # Finds the single nearest neighbor in the point cloud
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pad_world_pos, 1)

            # Retrieve the actual coordinate of the point in the cloud
            # idx[0] is the index of the nearest point
            nearest_surface_point = all_points[idx[0]]

            contact_points.append(nearest_surface_point)

        return contact_points


# --- Strategy Registry/Factory ---

STRATEGY_REGISTRY: Dict[str, Type[GraspContactStrategy]] = {
    "single": SinglePadStrategy,
    "projection": MultiPadProjectionStrategy,
    # Future strategies (e.g., "physical_adjust") can be added here
}
