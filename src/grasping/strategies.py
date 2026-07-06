import abc
import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Type

# Import BaseSuctionPad for type hinting
from src.grippers.pads import BaseSuctionPad


class GraspContactStrategy(abc.ABC):
    """
    Abstract Base Class for Grasp Contact Strategies.

    Responsibility:
    Receives a candidate Pose (TCP + Rotation) and determines WHERE
    each suction pad would touch the object's surface.
    """

    @abc.abstractmethod
    def resolve_contacts(
        self,
        tcp_pose: np.ndarray,
        pads: List[BaseSuctionPad],
        pcd_tree: o3d.geometry.KDTreeFlann,
        all_points: np.ndarray,
        max_pad_gap: float = 0.005,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Calculates the actual contact points on the surface for each pad.

        Args:
            tcp_pose: 4x4 homogeneous transformation matrix of the TCP.
            pads: List of Pad objects.
            pcd_tree: KDTree of the point cloud.
            all_points: (N, 3) array of cloud points.
            max_pad_gap: Maximum allowed gap (unused in Projection strategy,
                         but kept for interface consistency).

        Returns:
            Tuple containing:
            1. List[np.ndarray]: A list of (x, y, z) contact points.
            2. np.ndarray: The TCP pose (unchanged in Projection strategy).
        """
        pass


class SinglePadStrategy(GraspContactStrategy):
    """
    Strategy for Single Suction Cup Grippers.
    Contact is simply the TCP origin.
    """

    def resolve_contacts(
        self,
        tcp_pose: np.ndarray,
        pads: List[BaseSuctionPad],
        pcd_tree: o3d.geometry.KDTreeFlann,
        all_points: np.ndarray,
        max_pad_gap: float = 0.005,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        try:
            # The contact is the TCP origin itself
            tcp_origin = tcp_pose[:3, 3]

            # Returns: [Contact Points], Original Pose
            return [tcp_origin], tcp_pose
        except Exception:
            return []


class MultiPadProjectionStrategy(GraspContactStrategy):
    """
    Logic:
    1. Keeps the gripper rigid at the candidate pose.
    2. Calculates the theoretical world position of each pad.
    3. Finds the nearest neighbor on the surface for each pad.
    """

    def resolve_contacts(
        self,
        tcp_pose: np.ndarray,
        pads: List[BaseSuctionPad],
        pcd_tree: o3d.geometry.KDTreeFlann,
        all_points: np.ndarray,
        max_pad_gap: float = 0.005,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        try:
            contact_points = []
            R_tcp = tcp_pose[:3, :3]
            t_tcp = tcp_pose[:3, 3]

            for pad in pads:
                # 1. Rigid World Position
                pad_world_pos = t_tcp + (R_tcp @ pad.offset)

                # Safety: Check for NaN/Inf (prevents KD-Tree crash)
                if not np.all(np.isfinite(pad_world_pos)):
                    # print(f"[Warning] Invalid Pad Position calculated: {pad_world_pos}. Using TCP as fallback.")
                    contact_points.append(t_tcp)
                    continue

                # 2. Projection onto Surface (Nearest Neighbor)
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pad_world_pos, 1)
                
                if k > 0:
                    nearest_surface_point = all_points[idx[0]]
                    contact_points.append(nearest_surface_point)
                else:
                    # If no neighbor found, fallback to theoretical position
                    contact_points.append(pad_world_pos)

            # Returns: [Projected Points], Original Pose (Unchanged)
            return contact_points, tcp_pose
        except Exception:
            return []


# --- Strategy Registry ---

STRATEGY_REGISTRY: Dict[str, Type[GraspContactStrategy]] = {
    "single": SinglePadStrategy,
    "projection": MultiPadProjectionStrategy,
}
