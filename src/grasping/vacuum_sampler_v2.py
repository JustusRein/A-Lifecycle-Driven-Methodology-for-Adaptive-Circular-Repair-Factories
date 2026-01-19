import copy
import numpy as np
import open3d as o3d
import trimesh
from dataclasses import dataclass
from typing import List, Optional, TypedDict, Dict, Literal

from src.grasping.base_sampler import BaseGraspSampler, GraspCandidate
from src.grippers.vacuum_gripper_v2 import VacuumGripper
import src.utils.geometry_utils as gu

# --- Types & Config ---


class VacuumScoreDetails(TypedDict):
    """
    Detailed breakdown of the grasp score for debugging.
    """

    flatness: float  # Aggregated flatness score
    verticality: float  # Alignment with gravity/Z
    torque: float  # Stability score (distance to CoM)
    raw_angle_deg: float  # The actual angle deviation
    pad_scores: Dict[
        str, float
    ]  # Individual score for each pad (e.g., {'left': 1.0, 'right': 0.0})
    failure_reason: Optional[str]


@dataclass
class VacuumSamplerConfig:
    """
    Configuration for the Vacuum Grasp Sampler.
    """

    # Sampling Parameters
    num_samples: int = 200
    approach_distance: float = 0.10

    # --- Orientation Strategy ---
    # "uniform": Rotate gripper around the normal vector (for asymmetric grippers).
    # "fixed": Use only the aligned normal (for symmetric round grippers).
    rotation_strategy: Literal["uniform", "fixed"] = "uniform"

    # Rotation step size in degrees (e.g., test every 45 deg)
    rotation_step_deg: int = 45

    # Maximum rotation range (e.g., 180 for bilateral symmetry, 360 for full asymmetry)
    rotation_range_deg: int = 180

    # --- Thresholds ---
    max_curvature: float = 0.05
    max_angle_deg: float = 45.0
    min_score: float = 0.5

    # Maximum allowed gap between a pad and the surface (meters).
    # If a pad is floating more than this value, the grasp is invalid.
    max_pad_gap: float = 0.01

    # --- Weights ---
    weight_flatness: float = 0.40
    weight_verticality: float = 0.30
    weight_torque: float = 0.30

    # --- Score Aggregation ---
    # How to combine scores from multiple pads?
    # "min": Conservative. If one pad fails, everything fails.
    # "mean": Average. Good for dense arrays of suction cups.
    score_aggregation_method: Literal["min", "mean", "median"] = "min"


class VacuumGraspSampler(
    BaseGraspSampler[VacuumGripper, VacuumSamplerConfig, VacuumScoreDetails]
):
    """
    Sampler logic for Vacuum Grippers (Single or Multi-Pad).
    Decouples Point Discovery (Raycasting) from Orientation Generation (Strategy).
    """

    def sample_grasps(
        self, pcd: o3d.geometry.PointCloud
    ) -> List[GraspCandidate[VacuumScoreDetails]]:
        """
        Main Pipeline Entry Point.
        """
        self.clear_candidates()

        if not self._validate_input(pcd):
            return []

        # 1. Phase 1: Generation (Geometry + Collision)
        # Finds surface points and generates N orientations per point based on strategy.
        self._generate_candidates(pcd)

        print(f"[VacuumSampler] Phase 1: Generated {len(self.candidates)} candidates.")

        if not self.candidates:
            return []

        # 2. Phase 2: Evaluation (GSS - Grasp Stability Score)
        # Calculates physical scores (Flatness, Torque) for all pads.
        self._evaluate_candidates(pcd)

        # 3. Phase 3: Filtering & Sorting
        self.valid_candidates = [
            c for c in self.candidates if c.score >= self.config.min_score
        ]
        self.valid_candidates.sort(key=lambda x: x.score, reverse=True)

        print(f"[VacuumSampler] Result: {len(self.valid_candidates)} valid grasps.")
        return self.valid_candidates

    def _validate_input(self, pcd: o3d.geometry.PointCloud) -> bool:
        if not pcd.has_points():
            print("[VacuumSampler] Error: Empty Point Cloud.")
            return False
        # Ensure normals exist (crucial for orientation alignment)
        if not pcd.has_normals():
            pcd.estimate_normals()
        return True

    # =========================================================================
    # Phase 1: Generation (Raycasting + Orientation Strategy)
    # =========================================================================

    def _generate_candidates(self, pcd: o3d.geometry.PointCloud):
        """
        Orchestrates raycasting and delegates orientation generation.
        """
        # A. Setup Raycasting Scene
        scene = gu.create_raycasting_scene_from_hull(pcd)
        if scene is None:
            return

        # B. Create Rays (Inward from sphere)
        rays_list = self._create_rays_for_pcd(pcd)
        if not rays_list:
            return

        # C. Batch Cast
        rays_tensor = o3d.core.Tensor(np.array(rays_list), dtype=o3d.core.Dtype.Float32)
        results = scene.cast_rays(rays_tensor)
        t_hits = results["t_hit"].numpy()

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # D. Process Hits
        for i, t_dist in enumerate(t_hits):
            if np.isinf(t_dist):
                continue

            ray_origin = rays_list[i][:3]
            ray_dir = rays_list[i][3:]

            self._process_single_hit(t_dist, ray_origin, ray_dir, pcd, pcd_tree)

    def _create_rays_for_pcd(self, pcd: o3d.geometry.PointCloud) -> List[np.ndarray]:
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        center = (min_bound + max_bound) / 2
        max_dim = np.max(max_bound - min_bound)
        radius = (max_dim / 2) * 1.5

        return gu.generate_inward_rays_from_sphere(
            center, radius, self.config.num_samples
        )

    def _process_single_hit(self, t_dist, ray_origin, ray_dir, pcd, pcd_tree):
        """
        Takes a raw ray hit, snaps it to the surface, checks basic collision,
        and then generates MULTIPLE orientations based on strategy.
        """
        # 1. Snap to real surface
        hull_hit_point = ray_origin + (t_dist * ray_dir)
        real_point, real_normal = gu.get_nearest_point_in_cloud(
            hull_hit_point, pcd, pcd_tree
        )

        # Ensure normal points OUT
        if np.dot(ray_dir, real_normal) > 0:
            real_normal = -real_normal

        # 2. Check collision at the BASE pose (without rotation logic first)
        # This is a quick check to discard obviously bad points (like deep holes)
        if self._is_in_collision(real_point, real_normal):
            return

        # 3. Strategy: Generate Orientations
        # Decouples "Finding a point" from "Determining rotations"
        poses = self._generate_orientations(real_point, real_normal)

        # 4. Store all generated poses as candidates
        for pose in poses:
            candidate = GraspCandidate[VacuumScoreDetails](
                transform=pose,
                contact_point=real_point,
                approach_vector=real_normal,
                score=0.0,
                score_details=self._make_empty_score_details(),
            )
            self.candidates.append(candidate)

    def _generate_orientations(
        self, point: np.ndarray, normal: np.ndarray
    ) -> List[np.ndarray]:
        """
        Strategy Dispatcher: Generates valid poses based on the config strategy.
        Delegates math to geometry_utils.
        """
        # 1. Calculate Base Rotation (Z aligned with Normal)
        R_base = gu.get_rotation_matrix_between_vectors(np.array([0, 0, 1]), normal)

        # 2. Dispatch
        dispatch_map = {
            "fixed": self._strategy_fixed_orientation,
            "uniform": self._strategy_uniform_fan,
        }
        strategy_func = dispatch_map.get(
            self.config.rotation_strategy, self._strategy_fixed_orientation
        )
        return strategy_func(point, R_base)
        # if self.config.rotation_strategy == "fixed":
        #     return self._strategy_fixed_orientation(point, R_base)
        #
        # elif self.config.rotation_strategy == "uniform":
        #     return self._strategy_uniform_fan(point, R_base)

    def _strategy_fixed_orientation(
        self, point: np.ndarray, base_rotation: np.ndarray
    ) -> List[np.ndarray]:
        """
        Returns a single pose aligned with the normal. (Symmetric grippers).
        """
        pose = gu.create_pose_matrix(base_rotation, point)
        return [pose]

    def _strategy_uniform_fan(
        self, point: np.ndarray, base_rotation: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generates a 'fan' of rotations around the normal vector. (Asymmetric grippers).
        """
        # Delegate heavy math to Geometry Utils
        rot_matrices = gu.generate_z_rotation_fan(
            base_rotation=base_rotation,
            step_deg=self.config.rotation_step_deg,
            range_deg=self.config.rotation_range_deg,
        )

        # Convert all rotations into 4x4 poses
        poses = [gu.create_pose_matrix(R, point) for R in rot_matrices]
        return poses

    def _is_in_collision(self, point: np.ndarray, normal: np.ndarray) -> bool:
        # Checks collision using the Gripper's safety volume generation logic
        safety_mesh = self.gripper.generate_safety_collision_mesh(
            contact_point=point,
            surface_normal=normal,
            approach_distance=self.config.approach_distance,
        )
        return self.check_collision(safety_mesh)

    # =========================================================================
    # Phase 2: Evaluation (Multi-Pad GSS)
    # =========================================================================

    def _evaluate_candidates(self, pcd: o3d.geometry.PointCloud):
        """
        Calculates scores for all candidates.
        """
        all_points = np.asarray(pcd.points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # Optimize: Calculate global stats once
        com_xy, max_torque_arm = self._calculate_global_stats(pcd, all_points)

        for cand in self.candidates:
            self._evaluate_single_candidate(
                cand, pcd_tree, all_points, com_xy, max_torque_arm
            )

    def _evaluate_single_candidate(
        self, cand, pcd_tree, all_points, com_xy, max_torque_arm
    ):
        # 1. Verticality (Robot Pose)
        s_vert, angle_deg = self._calculate_verticality(cand)
        if s_vert == 0.0:
            self._mark_candidate_failed(cand, "bad_angle")
            return

        # 2. Torque (Global Stability)
        s_torque = self._calculate_torque(cand, com_xy, max_torque_arm)

        # 3. Multi-Pad Flatness (Local Sealing)
        s_flat, pad_scores_dict = self._evaluate_pads_flatness(
            cand, pcd_tree, all_points
        )

        if s_flat <= 0.0:
            self._mark_candidate_failed(cand, "bad_seal")
            # Store partial details for debug even if failed
            cand.score_details["pad_scores"] = pad_scores_dict
            return

        # 4. Final Weighted Score
        score = (
            (self.config.weight_flatness * s_flat)
            + (self.config.weight_verticality * s_vert)
            + (self.config.weight_torque * s_torque)
        )

        cand.score = score
        cand.score_details = VacuumScoreDetails(
            flatness=s_flat,
            verticality=s_vert,
            torque=s_torque,
            raw_angle_deg=angle_deg,
            pad_scores=pad_scores_dict,
        )

    def _evaluate_pads_flatness(self, cand, pcd_tree, all_points):
        """
        Iterates over all pads, projects them to surface, checks gaps, and calculates flatness.
        """
        pad_scores = []
        pad_details = {}

        # Extract Candidate Rotation (3x3)
        R_cand = cand.transform[:3, :3]
        TCP_pos = cand.contact_point

        for pad in self.gripper.config.pads:
            # A. Transform Pad Offset to World Frame
            # P_world = TCP + (R * Offset_local)
            world_pad_pos = TCP_pos + (R_cand @ pad.offset)

            # B. Find Nearest Point on Surface
            [k, idx, _] = pcd_tree.search_knn_vector_3d(world_pad_pos, 1)
            nearest_pt = all_points[idx[0]]

            # C. Gap Check (Critical for Multi-Pad)
            gap = np.linalg.norm(world_pad_pos - nearest_pt)
            if gap > self.config.max_pad_gap:
                # Pad is floating in air -> Fail
                pad_scores.append(0.0)
                pad_details[pad.name] = 0.0
                continue

            # D. Local Curvature Calculation
            # Heuristic: 1.5x the safety radius of the specific pad
            search_radius = pad.safety_radius * 1.5

            [k, idx_neighbors, _] = pcd_tree.search_radius_vector_3d(
                nearest_pt, search_radius
            )

            if k < 5:
                # Edge/Noise -> Fail
                pad_scores.append(0.0)
                pad_details[pad.name] = 0.0
                continue

            # PCA for Curvature
            neighbors = all_points[idx_neighbors, :]
            cov = np.cov(neighbors, rowvar=False)
            eigs = np.linalg.eigvalsh(cov)
            curvature = eigs[0] / (np.sum(eigs) + 1e-12)

            # Score 0.0 (Curved) to 1.0 (Flat)
            s_pad = 1.0 - (curvature / self.config.max_curvature)
            s_pad = np.clip(s_pad, 0.0, 1.0)

            pad_scores.append(s_pad)
            pad_details[pad.name] = s_pad

        # Aggregation
        aggregated_flatness = self._aggregate_pad_scores(pad_scores)

        return aggregated_flatness, pad_details

    def _aggregate_pad_scores(self, scores: List[float]) -> float:
        """
        Combines multiple pad scores based on the configuration method.
        """
        if not scores:
            return 0.0

        method = self.config.score_aggregation_method

        if method == "min":
            return min(scores)  # Safest
        elif method == "mean":
            return np.mean(scores)  # Tolerant
        elif method == "median":
            return np.median(scores)  # Robust
        else:
            return min(scores)  # Fallback

    # =========================================================================
    # Helpers
    # =========================================================================

    def _calculate_global_stats(self, pcd, all_points):
        center_of_mass = pcd.get_center()
        com_xy = center_of_mass[:2]
        dists = np.linalg.norm(all_points[:, :2] - com_xy, axis=1)
        max_torque = np.max(dists) if len(dists) > 0 else 1.0
        return com_xy, max_torque

    def _calculate_verticality(self, cand):
        z_axis = np.array([0, 0, 1])
        dot = np.dot(cand.approach_vector, z_axis)
        angle_deg = np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0)))

        if angle_deg > self.config.max_angle_deg:
            return 0.0, angle_deg

        s_vert = 1.0 - (angle_deg / self.config.max_angle_deg)
        return np.clip(s_vert, 0.0, 1.0), angle_deg

    def _calculate_torque(self, cand, com_xy, max_torque):
        grasp_xy = cand.contact_point[:2]
        dist = np.linalg.norm(grasp_xy - com_xy)
        s_torque = 1.0 - (dist / max_torque)
        return np.clip(s_torque, 0.0, 1.0)

    def _mark_candidate_failed(self, cand, reason: str):
        cand.score = 0.0
        details = self._make_empty_score_details()
        # Mark error in pad_scores for debug purposes
        details["failure_reason"] = reason
        cand.score_details = details

    @staticmethod
    def _make_empty_score_details() -> VacuumScoreDetails:
        return VacuumScoreDetails(
            flatness=0.0,
            verticality=0.0,
            torque=0.0,
            raw_angle_deg=0.0,
            pad_scores={},
            failure_reason=None,  # Default empty
        )

    # =========================================================================
    # Visualization
    # =========================================================================

    def visualize_grasp(
        self,
        pcd: o3d.geometry.PointCloud,
        grasp: GraspCandidate,
        show_safety_volume: bool = False,
    ):
        """
        Visualizes the grasp using the Gripper's geometry generation logic.
        """
        geometries = []

        # 1. Ghost Object
        pcd_copy = copy.deepcopy(pcd)
        pcd_copy.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_copy)

        # 2. Gripper Geometry
        if show_safety_volume:
            # Generate Safety Volume (Trimesh) -> Convert to O3D
            mesh_trimesh = self.gripper.generate_safety_collision_mesh(
                grasp.contact_point,
                grasp.approach_vector,
                self.config.approach_distance,
            )
            mesh_o3d = gu.trimesh_to_open3d(mesh_trimesh)
            mesh_o3d.paint_uniform_color([0, 0, 1])  # Solid Blue
        else:
            # Generate Visual Mesh (GenericGeometry) -> Extract O3D
            wrapper = self.gripper.generate_collision_mesh()
            mesh_o3d = copy.deepcopy(wrapper.geometry)

            # Transform from Origin to Grasp Pose
            mesh_o3d.transform(grasp.transform)

        geometries.append(mesh_o3d)

        # 3. Coordinate Frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(grasp.transform)
        geometries.append(frame)

        o3d.visualization.draw_geometries(
            geometries, window_name=f"Score: {grasp.score:.2f}"
        )

    def visualize_candidates_heatmap(
        self,
        pcd: o3d.geometry.PointCloud,
        attribute: str = "total",
        relative_scale: bool = False,
        valid_only: bool = True,
    ):
        """
        Visualizes candidates as a colored point cloud based on score.
        """
        cands = self.valid_candidates if valid_only else self.candidates
        if not cands:
            print("[Visualizer] No candidates.")
            return

        geometries = []
        # Background Ghost Object
        pcd_ghost = copy.deepcopy(pcd)
        pcd_ghost.paint_uniform_color([0.8, 0.8, 0.8])
        geometries.append(pcd_ghost)

        # Extract Values
        points = []
        raw_values = []

        for c in cands:
            points.append(c.contact_point)
            if attribute == "total":
                val = c.score
            else:
                val = c.score_details.get(attribute, 0.0)
            raw_values.append(val)

        # Coloring Logic
        if relative_scale and len(raw_values) > 0:
            v_min, v_max = min(raw_values), max(raw_values)
            span = v_max - v_min
            if span < 1e-6:
                final_scores = [1.0 for _ in raw_values]
            else:
                final_scores = [(v - v_min) / span for v in raw_values]
        else:
            final_scores = [max(0.0, min(1.0, v)) for v in raw_values]

        # Generate Heatmap
        heatmap_pcd = gu.create_score_heatmap_pcd(points, final_scores)
        geometries.append(heatmap_pcd)

        o3d.visualization.draw_geometries(
            geometries, window_name=f"Heatmap: {attribute.upper()}"
        )
