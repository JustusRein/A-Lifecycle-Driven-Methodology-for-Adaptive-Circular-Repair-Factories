import copy
import trimesh.geometry
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import List, TypedDict

from src.grasping.base_sampler import BaseGraspSampler, GraspCandidate
from src.grippers.vacuum_gripper import VacuumGripper
import src.utils.geometry_utils as gu


@dataclass
class VacuumSamplerConfig:
    # ... existing params ...
    num_samples: int = 200
    approach_distance: float = 0.10

    # Thresholds
    max_curvature: float = 0.05
    max_angle_deg: float = 45.0
    min_score: float = 0.5

    # GSS Weights (Sum should ideally be 1.0, but not strictly required)
    weight_flatness: float = 0.40  # Importance of sealing
    weight_verticality: float = 0.30  # Importance of robot pose
    weight_torque: float = 0.30  # Importance of stability


class VacuumScoreDetails(TypedDict):
    flatness: float
    verticality: float
    torque: float
    raw_angle_deg: float


class VacuumGraspSampler(
    BaseGraspSampler[VacuumGripper, VacuumSamplerConfig, VacuumScoreDetails]
):
    """
    Vacuum-specific sampler using Spherical Ray Casting + Snapping.
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
        self._find_potential_tcps_raycasting(pcd)
        print(
            f"[VacuumSampler] Phase 1: Found {len(self.candidates)} collision-free TCPs."
        )

        if not self.candidates:
            return []

        # 2. Phase 2: Evaluation (GSS)
        self._evaluate_gss(pcd)

        # 3. Filter & Sort (CRITICAL FIXES HERE)
        # Filtra apenas os bons
        self.valid_candidates = [
            c for c in self.candidates if c.score >= self.config.min_score
        ]

        # Ordena: MAIOR score primeiro (reverse=True)
        self.valid_candidates.sort(key=lambda x: x.score, reverse=True)

        print(
            f"[VacuumSampler] Result: {len(self.valid_candidates)} valid grasps "
            f"(from {len(self.candidates)} raw candidates)."
        )

        # Retorna a lista validada
        return self.valid_candidates

    def _validate_input(self, pcd: o3d.geometry.PointCloud) -> bool:
        if not pcd.has_points():
            print("[VacuumSampler] Error: Empty Point Cloud.")
            return False
        if not pcd.has_normals():
            pcd.estimate_normals()
        return True

    def _find_potential_tcps_raycasting(self, pcd: o3d.geometry.PointCloud):
        """
        Orchestrates the Ray Casting process.
        """
        # A. Setup Scene
        scene = gu.create_raycasting_scene_from_hull(pcd)
        if scene is None:
            return

        # B. Generate Rays
        rays_list = self._create_rays_for_pcd(pcd)
        if not rays_list:
            return

        # C. Cast Rays (Batch Operation)
        rays_tensor = o3d.core.Tensor(np.array(rays_list), dtype=o3d.core.Dtype.Float32)
        results = scene.cast_rays(rays_tensor)
        t_hits = results["t_hit"].numpy()

        # D. Process Results
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        for i, t_dist in enumerate(t_hits):
            if t_dist == float("inf"):
                continue

            # Extract Ray Data
            ray_origin = rays_list[i][:3]
            ray_dir = rays_list[i][3:]

            # Process single hit
            self._process_single_hit(t_dist, ray_origin, ray_dir, pcd, pcd_tree)

    def _create_rays_for_pcd(self, pcd: o3d.geometry.PointCloud) -> List[np.ndarray]:
        """
        Calculates bounding sphere and generates inward rays.
        """
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        center = (min_bound + max_bound) / 2
        max_dim = np.max(max_bound - min_bound)
        radius = (max_dim / 2) * 1.5

        return gu.generate_inward_rays_from_sphere(
            center, radius, self.config.num_samples
        )

    def _process_single_hit(
        self,
        t_dist: float,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        pcd: o3d.geometry.PointCloud,
        pcd_tree: o3d.geometry.KDTreeFlann,
    ):
        """
        Evaluates a single ray hit: Snaps to real surface, checks collision, adds candidate.
        """
        # 1. Calculate virtual hit on Convex Hull
        hull_hit_point = ray_origin + (t_dist * ray_dir)

        # 2. Snap to nearest REAL point on the cloud
        real_point, real_normal = gu.get_nearest_point_in_cloud(
            hull_hit_point, pcd, pcd_tree
        )
        if np.dot(ray_dir, real_normal) > 0:
            real_normal = -real_normal
        # 3. Check Collision (Gripper Body vs Environment)
        # Note: No orientation filters here. We defer quality checks to GSS Phase.
        if self._is_in_collision(real_point, real_normal):
            return

        # 4. Create and Store Candidate
        pose_matrix = self._calculate_pose(real_point, real_normal)

        candidate = GraspCandidate[VacuumScoreDetails](
            transform=pose_matrix,
            contact_point=real_point,
            approach_vector=real_normal,
            score=0.0,  # To be calculated in GSS
            score_details=self._make_empty_score_details(),
        )
        self.candidates.append(candidate)

    def _is_in_collision(self, point: np.ndarray, normal: np.ndarray) -> bool:
        """
        Helper to generate safety mesh and check collision.
        """
        safety_mesh = self.gripper.generate_safety_collision_mesh(
            contact_point=point,
            surface_normal=normal,
            approach_distance=self.config.approach_distance,
        )
        return self.check_collision(safety_mesh)

    def _calculate_pose(self, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
        # Relies on trimesh to align Z with Normal

        z_axis = np.array([0, 0, 1])
        T = trimesh.geometry.align_vectors(z_axis, normal)
        T[:3, 3] = point
        return T

    def get_best_candidates(
        self, n: int = 5
    ) -> List[GraspCandidate[VacuumScoreDetails]]:
        """
        Returns the top N candidates sorted by score (Highest first).
        """
        # Sort descending: Highest score -> Best grasp
        _n = min(n, len(self.valid_candidates))
        return self.valid_candidates[:_n]

    def visualize_grasp(
        self,
        pcd: o3d.geometry.PointCloud,
        grasp: GraspCandidate[VacuumScoreDetails],
        show_safety_volume: bool = False,
    ):
        """
        Opens a visualizer window showing the object + gripper at the grasp pose.

        Args:
            pcd: The object point cloud.
            grasp: The specific candidate to visualize.
            show_safety_volume:
                False (Default) = Show the realistic robot mesh (Blue/Grey).
                True = Show the safety collision cylinder (Solid Blue).
        """
        geometries = []

        # 1. Object (Cloned & Colored Grey)
        pcd_copy = copy.deepcopy(pcd)
        pcd_copy.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_copy)

        # 2. Gripper Geometry
        if show_safety_volume:
            # A. Safety Volume (Collision Tube)
            # Returns Trimesh -> Convert to Open3D
            mesh_trimesh = self.gripper.generate_safety_collision_mesh(
                grasp.contact_point,
                grasp.approach_vector,
                self.config.approach_distance,
            )
            mesh_o3d = gu.trimesh_to_open3d(mesh_trimesh)
            mesh_o3d.paint_uniform_color([0, 0, 1])  # Solid Blue
        else:
            # B. Visual Mesh (Realistic)
            # Returns GenericGeometry wrapper -> Get Open3D Mesh
            gripper_wrapper = self.gripper.generate_collision_mesh()
            mesh_o3d = gripper_wrapper.geometry

            # CRITICAL: Deepcopy so we don't move the original "master" mesh
            mesh_o3d = copy.deepcopy(mesh_o3d)

            # Move mesh from Origin (0,0,0) to the Grasp Pose
            mesh_o3d.transform(grasp.transform)

        geometries.append(mesh_o3d)

        # 3. TCP Coordinate Frame (RGB Axis)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(grasp.transform)
        geometries.append(frame)

        # 4. Render
        mode_str = "Safety Volume" if show_safety_volume else "Visual Mesh"
        window_name = f"Grasp Visualization [{mode_str}] - Score: {grasp.score:.3f}"

        o3d.visualization.draw_geometries(geometries, window_name=window_name)

    def visualize_candidates_heatmap(
        self,
        pcd: o3d.geometry.PointCloud,
        attribute: str = "total",
        relative_scale: bool = False,
        valid_only: bool = True,
    ):
        """
        Visualizes candidates colored by a specific score attribute.

        Args:
            pcd: The object point cloud.
            attribute: Which score to visualize?
                       "total" (default), "flatness", "verticality", "torque".
            relative_scale:
                False = Absolute (0.0 is Red, 1.0 is Green).
                True  = Relative (Min value in list is Red, Max value is Green).
        """
        cands = self.valid_candidates if valid_only else self.candidates
        if not cands:
            print("[Visualizer] No candidates.")
            return

        geometries = []

        # 1. Background Ghost Object
        pcd_ghost = copy.deepcopy(pcd)
        pcd_ghost.paint_uniform_color([0.8, 0.8, 0.8])
        geometries.append(pcd_ghost)

        # 2. Extract Values based on 'attribute'
        points = []
        raw_values = []

        for c in cands:
            points.append(c.contact_point)

            if attribute == "total":
                val = c.score
            else:
                # Safely get detail, default to 0.0 if missing (e.g. rejected points)
                val = c.score_details.get(attribute, 0.0)

            raw_values.append(val)

        # 3. Handle Scaling (Absolute vs Relative)
        if relative_scale and len(raw_values) > 0:
            v_min, v_max = min(raw_values), max(raw_values)
            span = v_max - v_min
            if span < 1e-6:  # Avoid division by zero if all scores are identical
                final_scores = [1.0 for _ in raw_values]  # All Green
            else:
                # Normalize min->0.0, max->1.0
                final_scores = [(v - v_min) / span for v in raw_values]
                print(
                    f"[Visualizer] Relative Mode: Mapping [{v_min:.3f}, {v_max:.3f}] -> [0, 1]"
                )
        else:
            # Absolute Mode (Clamp 0-1)
            final_scores = [max(0.0, min(1.0, v)) for v in raw_values]
        # 4. Generate Heatmap Cloud
        # We reuse the utility function we made earlier
        heatmap_pcd = gu.create_score_heatmap_pcd(points, final_scores)
        geometries.append(heatmap_pcd)

        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Heatmap: {attribute.upper()} (Relative={relative_scale})",
        )

    @staticmethod
    def _make_empty_score_details() -> VacuumScoreDetails:
        return VacuumScoreDetails(
            {
                "flatness": 0.0,
                "verticality": 0.0,
                "torque": 0.0,
                "raw_angle_deg": 0.0,
            }
        )

    def _evaluate_gss(self, pcd: o3d.geometry.PointCloud):
        """
        Phase 2: GSS Evaluation.
        Calculates score based on: Flatness, Verticality, and Torque Arm.
        """
        # --- 1. Pre-calculation of Global Stats ---
        # We calculate these once per cloud, not per candidate
        all_points = np.asarray(pcd.points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        com_xy, max_torque_arm = self._calculate_global_torque_stats(pcd, all_points)

        # --- 2. Candidate Evaluation ---
        for cand in self.candidates:
            # Factor 1: Verticality
            s_vert, angle_deg = self._calculate_verticality(cand)

            # Optimization: If hard filtered by verticality, skip heavy computations
            if s_vert == 0.0 and angle_deg > self.config.max_angle_deg:
                self._assign_zero_score(cand)
                continue

            # Factor 2: Flatness
            s_flat = self._calculate_flatness(cand, pcd_tree, all_points)
            if s_flat == 0.0:
                self._assign_zero_score(cand)
                continue

            # Factor 3: Torque
            s_torque = self._calculate_torque(cand, com_xy, max_torque_arm)

            # Final Weighted Score
            cand.score = (
                (self.config.weight_flatness * s_flat)
                + (self.config.weight_verticality * s_vert)
                + (self.config.weight_torque * s_torque)
            )

            cand.score_details = VacuumScoreDetails(
                flatness=s_flat,
                verticality=s_vert,
                torque=s_torque,
                raw_angle_deg=angle_deg,
            )

    def _calculate_global_torque_stats(self, pcd, all_points):
        """Calculates Center of Mass and the Max Radius for normalization."""
        center_of_mass = pcd.get_center()

        points_xy = all_points[:, :2]  # Drop Z
        com_xy = center_of_mass[:2]

        # Calculate distances of all points to CoM (XY only)
        dists = np.linalg.norm(points_xy - com_xy, axis=1)
        max_torque_arm = np.max(dists) if len(dists) > 0 else 1.0

        if max_torque_arm < 1e-6:
            max_torque_arm = 1.0

        return com_xy, max_torque_arm

    def _calculate_verticality(self, cand):
        """Calculates alignment with Z-axis."""
        z_axis = np.array([0, 0, 1])
        dot = np.dot(cand.approach_vector, z_axis)
        angle_rad = np.arccos(np.clip(abs(dot), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # Hard Filter
        if angle_deg > self.config.max_angle_deg:
            return 0.0, angle_deg

        # Linear Score
        s_vert = 1.0 - (angle_deg / self.config.max_angle_deg)
        return np.clip(s_vert, 0.0, 1.0), angle_deg

    def _calculate_flatness(self, cand, pcd_tree, all_points):
        """Calculates local surface curvature."""
        search_radius = self.gripper.config.cup_radius * 1.5

        [k, idx_neighbors, _] = pcd_tree.search_radius_vector_3d(
            cand.contact_point, search_radius
        )

        if k < 5:
            return 0.0

        neighbors = all_points[idx_neighbors, :]
        cov_matrix = np.cov(neighbors, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)

        min_eigen = eigenvalues[0]
        sum_eigen = np.sum(eigenvalues) + 1e-12
        curvature = min_eigen / sum_eigen

        # Linear Score
        s_flat = 1.0 - (curvature / self.config.max_curvature)
        return np.clip(s_flat, 0.0, 1.0)

    def _calculate_torque(self, cand, com_xy, max_torque_arm):
        """Calculates distance from grasp point to CoM on XY plane."""
        grasp_xy = cand.contact_point[:2]
        torque_arm = np.linalg.norm(grasp_xy - com_xy)

        # Linear Score
        s_torque = 1.0 - (torque_arm / max_torque_arm)
        return np.clip(s_torque, 0.0, 1.0)

    def _assign_zero_score(self, cand):
        """Helper to zero out a candidate."""
        cand.score = 0.0
        cand.score_details = self._make_empty_score_details()
