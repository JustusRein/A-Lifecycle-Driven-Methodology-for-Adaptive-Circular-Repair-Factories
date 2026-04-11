import copy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

import src.utils.geometry_utils as gu
from src.grasping.base_sampler import BaseGraspSampler, GraspCandidate
from src.grasping.strategies import STRATEGY_REGISTRY
from src.grippers.vacuum_gripper_v2 import VacuumGripper


# --- Types & Config ---
class RaycastingPatternDict(TypedDict, total=False):
    sphere: int
    top_down: int


class VacuumScoreDetails(TypedDict):
    """
    Detailed breakdown of the grasp score for debugging.
    """

    seal_score: float  # Aggregated seal score (coverage of contact zones)
    verticality: float  # Alignment with gravity/Z
    torque: float  # Stability score (distance to CoM)
    raw_angle_deg: float  # The actual angle deviation
    pad_scores: Dict[
        str, float
    ]  # Individual score for each pad (e.g., {'left': 1.0, 'right': 0.0})
    failure_reason: List[str]


@dataclass
class VacuumSamplerConfig:
    """
    Configuration for the Vacuum Grasp Sampler.
    """

    # Sampling Parameters
    raycasting_samples: RaycastingPatternDict = field(
        default_factory=lambda: {"sphere": 100, "top_down": 150}
    )
    # --- Raycasting & Mesh Generation ---
    raycasting_hull_type: Literal["convex", "bpa", "poisson", "alpha"] = "alpha"
    raycasting_hemisphere_only: bool = False

    # Distance in meters that the gripper will "reach into" the object during collision checks.
    approach_distance: float = 0.10
    # When checking for collisions, we ignore the first 1.5cm (0.015m) from the contact point to allow the suction cup to touch the surface without being considered a collision.
    local_clearence_margin: float = 0.015

    # Collision volume shape: "bounding_box_shape", "extended_body_shape", "gripper_shape"
    safety_volume_shape: Literal[
        "bounding_box_shape", "extended_body_shape", "gripper_shape"
    ] = "extended_body_shape"

    # --- Orientation Strategy ---
    # "uniform": Rotate gripper around the normal vector (for asymmetric grippers).
    # "fixed": Use only the aligned normal (for symmetric round grippers).
    rotation_strategy: Literal["uniform", "fixed"] = "uniform"

    # Rotation step size in degrees (e.g., test every 45 deg)
    rotation_step_deg: int = 12

    # Maximum rotation range (e.g., 180 for bilateral symmetry, 360 for full asymmetry)
    rotation_range_deg: int = 180

    # --- Thresholds ---
    max_angle_deg: float = 45.0
    min_seal_threshold: float = 0.5
    min_score: float = 0.5

    # --- Weights ---
    weight_seal: float = 0.35
    weight_verticality: float = 0.40
    weight_torque: float = 0.25

    # --- Score Aggregation ---
    # How to combine scores from multiple pads?
    # "min": Conservative. If one pad fails, everything fails.
    # "mean": Average. Good for dense arrays of suction cups.
    score_aggregation_method: Literal["min", "mean", "median"] = "min"
    debug_score: bool = (
        False  # Calculate all attributes even if it fails for final score
    )


class VacuumGraspSampler(
    BaseGraspSampler[VacuumGripper, VacuumSamplerConfig, VacuumScoreDetails]
):
    """
    Sampler logic for Vacuum Grippers (Single or Multi-Pad).
    Decouples Point Discovery (Raycasting) from Orientation Generation (Strategy).
    """

    def __init__(self, gripper: VacuumGripper, config: VacuumSamplerConfig):
        super().__init__(gripper, config)

        strategy_name = self.gripper.config.grasp_strategy
        if strategy_name not in STRATEGY_REGISTRY:
            print(
                f"[Warning] Strategy '{strategy_name}' not found. Fallback to 'projection'."
            )
            raise ValueError(
                f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}"
            )

        self.strategy = STRATEGY_REGISTRY[strategy_name]()
        print(f"[VacuumSampler] Using Contact Strategy: {strategy_name}")

    def sample_grasps(
        self, pcd: o3d.geometry.PointCloud
    ) -> List[GraspCandidate[VacuumScoreDetails]]:
        """
        Main Pipeline Entry Point.
        """
        self.clear_candidates()

        if not self._validate_input(pcd):
            return []
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01, max_nn=30
                )
            )

        # 1. Phase 1: Generation (Geometry + Collision)
        self._generate_candidates(pcd)

        print(f"[VacuumSampler] Phase 1: Generated {len(self.candidates)} candidates.")

        if not self.candidates:
            return []

        # 2. Phase 2: Evaluation (GSS - Grasp Stability Score)
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
        # A. Setup Raycasting Scene using the configured hull type
        scene = gu.create_raycasting_scene(
            pcd, hull_type=self.config.raycasting_hull_type
        )
        if scene is None:
            return

        # B. Create Rays (Inward from sphere / upper hemisphere)
        rays_list = self._create_rays_for_pcd(pcd)
        if not rays_list:
            return

        # C. Batch Cast
        rays_tensor = o3d.core.Tensor(np.array(rays_list), dtype=o3d.core.Dtype.Float32)
        results = scene.cast_rays(rays_tensor)
        t_hits = results["t_hit"].numpy()

        # >>> CHAMADA DE DEBUG <<< (Pode comentar depois se quiser silenciar)
        max_dim = np.max(pcd.get_max_bound() - pcd.get_min_bound())
        self._debug_raycasting(pcd, rays_list, t_hits, radius=(max_dim / 2) * 1.5)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # D. Process Hits
        for i, t_dist in enumerate(t_hits):
            if np.isinf(t_dist):
                continue

            ray_origin = rays_list[i][:3]
            ray_dir = rays_list[i][3:]

            self._process_single_hit(t_dist, ray_origin, ray_dir, pcd, pcd_tree)

    def _create_rays_for_pcd(self, pcd: o3d.geometry.PointCloud) -> List[np.ndarray]:
        rays_list = []

        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()

        samples_sphere = self.config.raycasting_samples.get("sphere", 0)
        samples_top = self.config.raycasting_samples.get("top_down", 0)

        # --- 1. Sphere ---
        if samples_sphere > 0:
            center = (min_bound + max_bound) / 2
            max_dim = np.max(max_bound - min_bound)
            radius = (max_dim / 2) * 1.5

            sphere_rays = gu.generate_inward_rays_from_sphere(
                center,
                radius,
                samples_sphere,
                hemisphere_only=self.config.raycasting_hemisphere_only,
            )
            rays_list.extend(sphere_rays)

        # --- 2. Top-Down ---
        if samples_top > 0:
            top_down_rays = gu.generate_top_down_rays(
                min_bound=min_bound,
                max_bound=max_bound,
                num_samples=samples_top,
            )
            rays_list.extend(top_down_rays)

        return rays_list

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

        # 2. Check collision at the BASE pose
        if self._is_in_collision(
            real_point, real_normal, pcd, self.config.safety_volume_shape
        ):
            return

        # 3. Strategy: Generate Orientations
        poses = self._generate_orientations(real_point, real_normal)

        # 4. Store all generated poses as candidates
        for pose in poses:
            candidate = GraspCandidate[VacuumScoreDetails](
                transform=pose,
                contact_point=real_point,
                approach_vector=-real_normal,
                score=0.0,
                score_details=self._make_empty_score_details(),
            )
            self.candidates.append(candidate)

    def _generate_orientations(
        self, point: np.ndarray, normal: np.ndarray
    ) -> List[np.ndarray]:
        R_base = gu.get_rotation_matrix_between_vectors(np.array([0, 0, 1]), normal)

        dispatch_map = {
            "fixed": self._strategy_fixed_orientation,
            "uniform": self._strategy_uniform_fan,
        }
        strategy_func = dispatch_map.get(
            self.config.rotation_strategy, self._strategy_fixed_orientation
        )
        return strategy_func(point, R_base)

    def _strategy_fixed_orientation(
        self, point: np.ndarray, base_rotation: np.ndarray
    ) -> List[np.ndarray]:
        pose = gu.create_pose_matrix(base_rotation, point)
        return [pose]

    def _strategy_uniform_fan(
        self, point: np.ndarray, base_rotation: np.ndarray
    ) -> List[np.ndarray]:
        rot_matrices = gu.generate_z_rotation_fan(
            base_rotation=base_rotation,
            step_deg=self.config.rotation_step_deg,
            range_deg=self.config.rotation_range_deg,
        )
        poses = [gu.create_pose_matrix(R, point) for R in rot_matrices]
        return poses

    # ====================================================================
    # COLLISION SHAPE SUB-METHODS
    # ====================================================================

    def _check_bounding_box_collision(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        pcd: o3d.geometry.PointCloud,
        start_offset: float,
        total_length: float,
        body_radius: float,
        **kwargs,
    ) -> bool:
        center = point + normal * (start_offset + total_length / 2.0)
        extent = np.array([body_radius * 2, body_radius * 2, total_length])
        R = gu.get_rotation_matrix_between_vectors(np.array([0, 0, 1]), normal)

        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        cropped_pcd = pcd.crop(obb)

        return len(cropped_pcd.points) > 5

    def _check_extended_body_collision(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        pcd: o3d.geometry.PointCloud,
        start_offset: float,
        total_length: float,
        body_radius: float,
        **kwargs,
    ) -> bool:
        pts = np.asarray(pcd.points)
        vecs = pts - point

        z_distances = np.dot(vecs, normal)
        z_mask = (z_distances >= start_offset) & (
            z_distances <= start_offset + total_length
        )
        pts_in_z_slice = vecs[z_mask]

        if len(pts_in_z_slice) == 0:
            return False

        z_distances_filtered = z_distances[z_mask]
        radial_vecs = pts_in_z_slice - np.outer(z_distances_filtered, normal)
        radial_distances_sq = np.sum(radial_vecs**2, axis=1)

        points_inside_cylinder = np.sum(radial_distances_sq <= (body_radius**2))
        return points_inside_cylinder > 5

    def _check_gripper_shape_collision(
        self, pcd: o3d.geometry.PointCloud, safety_mesh, **kwargs
    ) -> bool:
        pts = np.asarray(pcd.points)
        try:
            inside_mask = safety_mesh.contains(pts)
            return np.sum(inside_mask) > 5
        except Exception as e:
            print(f"[Warning] Failed to run 'contains' on gripper mesh. Error: {e}")
            return False

    # ====================================================================
    # MAIN COLLISION METHOD
    # ====================================================================

    def _is_in_collision(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        pcd: o3d.geometry.PointCloud,
        safety_volume_shape: Literal[
            "gripper_shape", "bounding_box_shape", "extended_body_shape"
        ] = "bounding_box_shape",
    ) -> bool:
        # 1. GLOBAL SHIELD (Environment via Trimesh)
        safety_mesh = self.gripper.generate_safety_collision_mesh(
            contact_point=point,
            surface_normal=normal,
            approach_distance=self.config.approach_distance,
        )

        if self.check_collision(safety_mesh):
            return True

        # 2. LOCAL SHIELD SETUP (Target Object via Point Cloud)
        if self.config.approach_distance <= 0:
            return False

        clearance_margin = self.config.local_clearence_margin
        pad_length = self.gripper.config.pad_height
        approach_distance = self.config.approach_distance

        start_offset = clearance_margin
        total_length = pad_length + approach_distance
        body_radius = self.gripper.config.body_radius * (
            1.0 + self.gripper.config.collision_margin
        )

        shape_dispatcher = {
            "bounding_box_shape": self._check_bounding_box_collision,
            "extended_body_shape": self._check_extended_body_collision,
            "gripper_shape": self._check_gripper_shape_collision,
        }

        collision_method = shape_dispatcher.get(safety_volume_shape)

        if not collision_method:
            raise ValueError(
                f"Unknown safety_volume_shape: '{safety_volume_shape}'. Expected one of {list(shape_dispatcher.keys())}"
            )

        return collision_method(
            point=point,
            normal=normal,
            pcd=pcd,
            start_offset=start_offset,
            total_length=total_length,
            body_radius=body_radius,
            safety_mesh=safety_mesh,
        )

    # =========================================================================
    # Phase 2: Evaluation (Multi-Pad GSS)
    # =========================================================================

    def _evaluate_candidates(self, pcd: o3d.geometry.PointCloud):
        all_points = np.asarray(pcd.points)
        if not pcd.has_normals():
            pcd.estimate_normals()
        all_normals = np.asarray(pcd.normals)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        com_xy, max_torque_arm = self._calculate_global_stats(pcd, all_points)

        # Calculate the resolution (virtual voxel size) once per point cloud
        distances = pcd.compute_nearest_neighbor_distance()
        avg_point_spacing = np.mean(np.asarray(distances))
        for cand in self.candidates:
            self._evaluate_single_candidate(
                cand,
                pcd_tree,
                all_points,
                all_normals,
                com_xy,
                max_torque_arm,
                avg_point_spacing,
            )

    def debug_candidates(self, pcd: o3d.geometry.PointCloud, debug_idx: List[int]):
        all_points = np.asarray(pcd.points)
        if not pcd.has_normals():
            pcd.estimate_normals()
        all_normals = np.asarray(pcd.normals)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        com_xy, max_torque_arm = self._calculate_global_stats(pcd, all_points)

        for idx in debug_idx:
            cand = self.candidates[idx]
            self._evaluate_single_candidate(
                cand,
                pcd_tree,
                all_points,
                all_normals,
                com_xy,
                max_torque_arm,
                debug=True,
            )

    def _evaluate_single_candidate(
        self,
        cand: GraspCandidate[VacuumScoreDetails],
        pcd_tree: o3d.geometry.KDTreeFlann,
        all_points: np.ndarray,
        all_normals: np.ndarray,
        com_xy: np.ndarray,
        max_torque_arm: float,
        avg_point_spacing: float,
        debug: bool = False,
    ) -> None:
        fail_reasons: List[str] = []
        is_failed = False

        s_vert, angle_deg = self._calculate_verticality(cand)
        if s_vert == 0.0:
            is_failed = True
            fail_reasons.append("bad_angle")
            if not self.config.debug_score:
                self._mark_candidate_failed(cand, "bad_angle")
                return

        s_torque = self._calculate_torque(cand, com_xy, max_torque_arm)

        contact_points, adjusted_pose = self.strategy.resolve_contacts(
            cand.transform, self.gripper.config.pads, pcd_tree, all_points
        )

        cand.transform = adjusted_pose
        cand.contact_point = adjusted_pose[:3, 3]

        s_seal, pad_scores_dict = self._evaluate_suction_seal(
            cand,
            contact_points,
            pcd_tree,
            all_points,
            all_normals,
            debug,
            avg_point_spacing,
        )

        if s_seal <= self.config.min_seal_threshold:
            is_failed = True
            fail_reasons.append("bad_seal")
            if not self.config.debug_score:
                self._mark_candidate_failed(cand, "bad_seal")
                cand.score_details["pad_scores"] = pad_scores_dict
                return

        weighted_score = (
            (self.config.weight_seal * s_seal)
            + (self.config.weight_verticality * s_vert)
            + (self.config.weight_torque * s_torque)
        )

        cand.score = 0.0 if is_failed else weighted_score

        cand.score_details = VacuumScoreDetails(
            seal_score=s_seal,
            verticality=s_vert,
            torque=s_torque,
            raw_angle_deg=angle_deg,
            pad_scores=pad_scores_dict,
            failure_reason=fail_reasons,
        )

    def _evaluate_suction_seal(
        self,
        cand,
        contact_points,
        pcd_tree,
        all_points,
        all_normals,
        debug,
        avg_point_spacing,
    ):
        pad_scores = []
        pad_details = {}

        R_cand = cand.transform[:3, :3]
        TCP_pos = cand.contact_point

        # 1. Define the global baseline using the actual point spacing of the cloud
        global_density = 1.0 / (avg_point_spacing**2)

        for i, pad in enumerate(self.gripper.config.pads):
            nearest_pt = contact_points[i]

            world_pad_pos = TCP_pos + (R_cand @ pad.offset)
            gap = np.linalg.norm(world_pad_pos - nearest_pt)

            if gap > pad.max_sealing_distance:
                pad_scores.append(0.0)
                pad_details[pad.name] = 0.0
                continue

            search_radius = pad.safety_radius * 1.05
            [k, idx_neighbors, _] = pcd_tree.search_radius_vector_3d(
                nearest_pt, search_radius
            )

            # --- COARSE FILTER (Early-Reject) ---
            search_area = np.pi * (search_radius**2)
            expected_total_points = global_density * search_area

            # If it found less than 15% of the expected points based on the global baseline, fail immediately
            if k < (0.05 * expected_total_points):
                pad_scores.append(0.0)
                pad_details[pad.name] = 0.0
                continue

            neighbors = all_points[idx_neighbors, :]

            local_points_3d = (neighbors - world_pad_pos) @ R_cand
            filtered_local_points = local_points_3d[
                np.abs(local_points_3d[:, 2]) <= pad.max_sealing_distance
            ]

            zones_dict = pad.split_points_in_zones(filtered_local_points)
            if debug:
                debug_pad_projection(pad, filtered_local_points, zones_dict)

            zones_areas = pad.zone_areas
            zone_scores = []

            # --- FINE FILTER (Zone Evaluation) ---
            for zone_name, pts in zones_dict.items():
                target_area = zones_areas.get(zone_name, 0.0)

                # The expected point count requires the global density, not the local one
                expected_count = global_density * target_area
                valid_count = len(pts)
                print(
                    f"[Debug Sealing] Expected: {expected_count:.1f} | Valid: {valid_count} | Spacing: {avg_point_spacing:.4f}"
                )
                if valid_count == 0:
                    zone_scores.append(0.0)
                    continue

                ratio = valid_count / expected_count
                zone_scores.append(np.clip(ratio, 0, 1.0))

            current_pad_score = min(zone_scores) if zone_scores else 0.0
            pad_scores.append(current_pad_score)
            pad_details[pad.name] = current_pad_score

        # Aggregate pad scores using the configured method (min, mean, median)
        aggregated_seal = self._aggregate_pad_scores(pad_scores)

        return aggregated_seal, pad_details

    # def _evaluate_suction_seal(
    #     self, cand, contact_points, pcd_tree, all_points, all_normals, debug
    # ):
    #     pad_scores = []
    #     pad_details = {}
    #
    #     R_cand = cand.transform[:3, :3]
    #     TCP_pos = cand.contact_point
    #
    #     for i, pad in enumerate(self.gripper.config.pads):
    #         nearest_pt = contact_points[i]
    #
    #         world_pad_pos = TCP_pos + (R_cand @ pad.offset)
    #         gap = np.linalg.norm(world_pad_pos - nearest_pt)
    #
    #         if gap > pad.max_sealing_distance:
    #             pad_scores.append(0.0)
    #             pad_details[pad.name] = 0.0
    #             continue
    #
    #         search_radius = pad.safety_radius * 1.2
    #         [k, idx_neighbors, _] = pcd_tree.search_radius_vector_3d(
    #             nearest_pt, search_radius
    #         )
    #
    #         if k < 30:
    #             pad_scores.append(0.0)
    #             pad_details[pad.name] = 0.0
    #             continue
    #
    #         neighbors = all_points[idx_neighbors, :]
    #         ref_area = np.pi * (search_radius**2)
    #         local_density_ref = k / ref_area
    #
    #         local_points_3d = (neighbors - world_pad_pos) @ R_cand
    #         filtered_local_points = local_points_3d[
    #             np.abs(local_points_3d[:, 2]) <= pad.max_sealing_distance
    #         ]
    #
    #         zones_dict = pad.split_points_in_zones(filtered_local_points)
    #         if debug:
    #             debug_pad_projection(pad, filtered_local_points, zones_dict)
    #
    #         zones_areas = pad.zone_areas
    #         zone_scores = []
    #
    #         for zone_name, pts in zones_dict.items():
    #             target_area = zones_areas.get(zone_name, 0.0)
    #             expected_count = local_density_ref * target_area
    #             valid_count = len(pts)
    #             if valid_count == 0:
    #                 zone_scores.append(0.0)
    #                 continue
    #
    #             ratio = valid_count / expected_count
    #             zone_scores.append(np.clip(ratio, 0, 1.0))
    #
    #         current_pad_score = min(zone_scores) if zone_scores else 0.0
    #         pad_scores.append(current_pad_score)
    #         pad_details[pad.name] = current_pad_score
    #
    #     aggregated_seal = self._aggregate_pad_scores(pad_scores)
    #     aggregated_seal = np.min(pad_scores)
    #
    #     return aggregated_seal, pad_details

    def _aggregate_pad_scores(self, scores: List[float]) -> float:
        if not scores:
            return 0.0

        method = self.config.score_aggregation_method

        if method == "min":
            return min(scores)
        elif method == "mean":
            return np.mean(scores)
        elif method == "median":
            return np.median(scores)
        else:
            return min(scores)

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
        normal = -cand.approach_vector
        dot = np.dot(normal, z_axis)
        angle_deg = np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0)))

        decay_factor = 0.6 if dot < 0 else 1.0

        if angle_deg > self.config.max_angle_deg:
            # print("[Debug] Candidate failed verticality check: angle_deg =", angle_deg)
            return 0.0, angle_deg

        s_vert = 1.0 - (angle_deg / self.config.max_angle_deg)
        return decay_factor * (np.clip(s_vert, 0.0, 1.0)), angle_deg

    def _calculate_torque(self, cand, com_xy, max_torque):
        grasp_xy = cand.contact_point[:2]
        dist = np.linalg.norm(grasp_xy - com_xy)
        s_torque = 1.0 - (dist / max_torque)
        return np.clip(s_torque, 0.0, 1.0)

    def _mark_candidate_failed(self, cand, reason: str):
        cand.score = 0.0
        details = self._make_empty_score_details()
        details["failure_reason"] = [reason]
        cand.score_details = details

    @staticmethod
    def _make_empty_score_details() -> VacuumScoreDetails:
        return VacuumScoreDetails(
            seal_score=0.0,
            verticality=0.0,
            torque=0.0,
            raw_angle_deg=0.0,
            pad_scores={},
            failure_reason=[],
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
        geometries = []

        pcd_copy = copy.deepcopy(pcd)
        pcd_copy.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_copy)

        if show_safety_volume:
            body_direction = grasp.transform[:3, 2]

            mesh_trimesh = self.gripper.generate_safety_collision_mesh(
                contact_point=grasp.contact_point,
                surface_normal=body_direction,
                approach_distance=self.config.approach_distance,
            )

            mesh_o3d = gu.trimesh_to_open3d(mesh_trimesh)
            mesh_o3d.paint_uniform_color([0, 0, 1])
        else:
            wrapper = self.gripper.generate_collision_mesh()
            raw_geom = wrapper.geometry

            if isinstance(raw_geom, trimesh.Trimesh):
                mesh_o3d = gu.trimesh_to_open3d(raw_geom)
            else:
                mesh_o3d = copy.deepcopy(raw_geom)

            mesh_o3d.transform(grasp.transform)

        geometries.append(mesh_o3d)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(grasp.transform)
        geometries.append(frame)

        o3d.visualization.draw_geometries(
            geometries, window_name=f"Grasp Visualization - Score: {grasp.score:.3f}"
        )

    def visualize_candidates_heatmap(
        self,
        pcd: o3d.geometry.PointCloud,
        attribute: str = "total",
        relative_scale: bool = False,
        valid_only: bool = True,
    ):
        cands = self.valid_candidates if valid_only else self.candidates
        if not cands:
            print("[Visualizer] No candidates.")
            return

        geometries = []
        pcd_ghost = copy.deepcopy(pcd)
        pcd_ghost.paint_uniform_color([0.8, 0.8, 0.8])
        geometries.append(pcd_ghost)

        points = []
        raw_values = []

        for c in cands:
            points.append(c.contact_point)
            if attribute == "total":
                val = c.score
            else:
                val = c.score_details.get(attribute, 0.0)
            raw_values.append(val)

        if relative_scale and len(raw_values) > 0:
            v_min, v_max = min(raw_values), max(raw_values)
            span = v_max - v_min
            if span < 1e-6:
                final_scores = [1.0 for _ in raw_values]
            else:
                final_scores = [(v - v_min) / span for v in raw_values]
        else:
            final_scores = [max(0.0, min(1.0, v)) for v in raw_values]

        heatmap_pcd = gu.create_score_heatmap_pcd(points, final_scores)
        geometries.append(heatmap_pcd)

        o3d.visualization.draw_geometries(
            geometries, window_name=f"Heatmap: {attribute.upper()}"
        )

    def debug_specific_grasp(
        self, pcd: o3d.geometry.PointCloud, candidate: GraspCandidate
    ):
        all_points = np.asarray(pcd.points)
        all_normals = np.asarray(pcd.normals)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        com_xy, max_torque_arm = self._calculate_global_stats(pcd, all_points)

        print(f"--- Debugging Grasp at {candidate.contact_point} ---")
        self._evaluate_single_candidate(
            candidate, pcd_tree, all_points, all_normals, com_xy, max_torque_arm, True
        )

        self.visualize_grasp(pcd, candidate, show_safety_volume=True)

    def _debug_raycasting(self, pcd, rays_list, t_hits, radius):
        """
        Visualizador espetacular para o Raycasting!
        Mostra a nuvem, os raios, os pontos de impacto e a malha gerada (seja bpa, alpha, etc).
        """
        import copy

        geometries = []

        # 1. A Nuvem Original (Cinza escuro)
        pcd_vis = copy.deepcopy(pcd)
        pcd_vis.paint_uniform_color([0.4, 0.4, 0.4])
        geometries.append(pcd_vis)

        # 2. Em vez do Convex Hull, chama a geração real do hull que você configurou para vermos a verdade!
        scene = gu.create_raycasting_scene(
            pcd, hull_type=self.config.raycasting_hull_type, debug_visualize=False
        )

        # 3. Os Raios (Verde se acertou, Vermelho se passou reto)
        points = []
        lines = []
        colors = []
        hit_points = []

        for i, t in enumerate(t_hits):
            origin = rays_list[i][:3]
            dir_vec = rays_list[i][3:]

            if np.isinf(t):  # Raio perdeu-se no espaço
                end_pt = origin + dir_vec * (radius * 1.5)
                color = [0.8, 0.2, 0.2]  # Vermelho
            else:  # Raio bateu na malha
                end_pt = origin + dir_vec * t
                color = [0.2, 0.8, 0.2]  # Verde
                hit_points.append(end_pt)

            idx = len(points)
            points.extend([origin, end_pt])
            lines.append([idx, idx + 1])
            colors.append(color)

        ray_lines = o3d.geometry.LineSet()
        ray_lines.points = o3d.utility.Vector3dVector(points)
        ray_lines.lines = o3d.utility.Vector2iVector(lines)
        ray_lines.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(ray_lines)

        # 4. Os pontos de impacto exatos (Bolinhas Azuis)
        if hit_points:
            hits_pcd = o3d.geometry.PointCloud()
            hits_pcd.points = o3d.utility.Vector3dVector(hit_points)
            hits_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            geometries.append(hits_pcd)

        print(
            f"[Debug] Abrindo Visualizador de Raycasting ({self.config.raycasting_hull_type})..."
        )
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Raycasting Debug - {self.config.raycasting_hull_type}",
        )


def debug_pad_projection(pad, local_points_3d, zone_dict):
    """
    Visualizes the 2D projection of points in the suction cup's local frame.
    """
    plt.figure(figsize=(6, 6))

    plt.scatter(
        local_points_3d[:, 0],
        local_points_3d[:, 1],
        c="gray",
        s=1,
        alpha=0.5,
        label="All points",
    )

    colors = ["red", "green", "blue", "yellow"]
    for i, (zone_name, pts) in enumerate(zone_dict.items()):
        if len(pts) > 0:
            plt.scatter(pts[:, 0], pts[:, 1], s=5, label=f"Zone: {zone_name}")

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(
        pad.safety_radius * np.cos(theta),
        pad.safety_radius * np.sin(theta),
        "k--",
        label="External Radius",
    )
    if hasattr(pad, "inner_radius"):
        plt.plot(
            pad.inner_radius * np.cos(theta),
            pad.inner_radius * np.sin(theta),
            "r:",
            label="Internal Hole",
        )

    plt.title(f"Debug Projection: {pad.name}")
    plt.xlabel("Local X (m)")
    plt.ylabel("Local Y (m)")
    plt.axis("equal")
    plt.legend()
    plt.show()
