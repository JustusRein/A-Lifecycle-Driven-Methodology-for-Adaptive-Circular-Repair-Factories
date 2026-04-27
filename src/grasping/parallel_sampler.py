import copy
import numpy as np
import open3d as o3d
import cv2
import random
from math import acos, degrees
from sklearn.decomposition import PCA
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely import set_precision
from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Optional, Dict, Any

from src.grasping.base_sampler import BaseGraspSampler, GraspCandidate
from src.grippers.parallel_gripper import ParallelGripper


class ParallelScoreDetails(TypedDict):
    area_score: float
    center_score: float
    total_area: float


@dataclass
class ParallelSamplerConfig:
    # Tolerance and distance parameters
    plane_angle_thresh: float = 5.0
    min_remaining_points: int = 50
    min_points_per_plane: int = 50  # Used as fallback if dynamic calculation fails
    distance_threshold: float = 0.001  # 1mm in meters
    max_planes: int = 200
    margin_points_between_planes: float = 0.001

    # Drawing and visualization
    plt_graphic_padding: float = 0.01
    contour_image_padding: int = 10

    # Success criteria
    min_score: float = 0.3

    # ==========================================
    # Visualization Flags (For Debugging)
    # ==========================================
    no_image: bool = False
    show_all_planes_and_normals: bool = True
    show_planes_parallel_clustering: bool = True
    show_plane_pairs: bool = True
    show_plane_pair_and_proj_in_pcd: bool = True
    show_proj_pts_p1: bool = True
    show_proj_pts_p2: bool = True
    show_proj_pts_p3: bool = True
    show_proj_pts_p4: bool = True
    show_proj_pts_p5: bool = True


class ParallelGraspSampler(
    BaseGraspSampler[ParallelGripper, ParallelSamplerConfig, ParallelScoreDetails]
):
    """
    Sampler for Parallel Jaw Grippers.
    Structured, typed, and documented implementation of the original main_script.py logic.
    """

    def __init__(
        self, gripper: ParallelGripper, config: Optional[ParallelSamplerConfig] = None
    ):
        if config is None:
            config = ParallelSamplerConfig()
        super().__init__(gripper, config)

    def sample_grasps(self, pcd: o3d.geometry.PointCloud) -> List[GraspCandidate]:
        """
        Main method coordinating the grasp extraction pipeline.
        """
        self.clear_candidates()
        cfg = self.config
        g_cfg = self.gripper.config

        # =================================================================
        # DYNAMIC CALCULATION OF MIN_POINTS_PER_PLANE
        # =================================================================
        distances = np.asarray(pcd.compute_nearest_neighbor_distance())
        avg_dist = np.mean(distances) if len(distances) > 0 else 0.001
        area_per_point = avg_dist**2

        # Minimum physical area (Area of the Franka finger tip)
        finger_area = g_cfg.z_pg * g_cfg.b_pg
        min_required_area = finger_area * 0.5

        dynamic_min_points = int(min_required_area / area_per_point)
        dynamic_min_points = np.clip(dynamic_min_points, 50, len(pcd.points) // 10)

        print(f"🧠 [Auto-Tuning] Average distance: {avg_dist:.4f}m")
        print(
            f"🧠 [Auto-Tuning] Dynamic limit defined: {dynamic_min_points} points/plane"
        )

        # Variables derived from gripper kinematics
        y_pg = max(
            g_cfg.q_pg + 2 * g_cfg.r_pg,
            g_cfg.h_pg + 2 * g_cfg.k_pg,
            g_cfg.f_pg + 2 * (g_cfg.a_pg + g_cfg.v_pg),
        )
        rd = max(g_cfg.ra, g_cfg.rb)

        # =================================================================
        # 1. PLANE SEGMENTATION AND MERGING
        # =================================================================
        print(
            f"🚀 Initiating plane extraction (Min pts/plane: {dynamic_min_points})..."
        )
        pcd_target = copy.deepcopy(pcd)
        original_points = np.asarray(pcd_target.points)
        original_indices = np.arange(len(original_points))

        plane_indices_list: List[np.ndarray] = []
        plane_colors: List[List[float]] = []
        plane_models: List[np.ndarray] = []
        plane_normals: List[np.ndarray] = []

        rest_pcd = copy.deepcopy(pcd_target)

        # 1.1 Raw Extraction (Classic RANSAC)
        for _ in range(cfg.max_planes):
            if len(rest_pcd.points) < cfg.min_remaining_points:
                break

            try:
                plane_model, inliers = rest_pcd.segment_plane(
                    distance_threshold=cfg.distance_threshold,
                    ransac_n=3,
                    num_iterations=1000,
                )
            except RuntimeError:
                break

            # CORE LOGIC: Stop extraction as soon as the found plane is junk/too small
            if len(inliers) < dynamic_min_points:
                break

            original_idx = original_indices[inliers]
            plane_indices_list.append(original_idx)

            plane_models.append(np.asarray(plane_model))
            normal_vector = np.asarray(plane_model[0:3])
            plane_normals.append(normal_vector / np.linalg.norm(normal_vector))

            plane_colors.append([random.random(), random.random(), random.random()])

            # Update remaining cloud
            rest_pcd = rest_pcd.select_by_index(inliers, invert=True)
            original_indices = np.delete(original_indices, inliers)

        print(f"   -> RANSAC extracted {len(plane_models)} raw planes.")

        # 1.2 Coplanar Plane Merging
        all_points_xyz = np.asarray(pcd_target.points)
        plane_models, plane_normals, plane_indices_list = self._merge_coplanar_planes(
            p_models=plane_models,
            p_normals=plane_normals,
            p_indices=plane_indices_list,
            all_pts=all_points_xyz,
            angle_thresh_deg=cfg.plane_angle_thresh,
            offset_thresh=cfg.distance_threshold * 2,
        )

        print(
            f"   -> After merging, {len(plane_normals)} consolidated structural planes remain."
        )

        if cfg.show_all_planes_and_normals and not cfg.no_image:
            colored_pcd = copy.deepcopy(pcd_target)
            colors_vis = np.ones((len(original_points), 3)) * [0.5, 0.5, 0.5]
            for indices, color in zip(plane_indices_list, plane_colors):
                colors_vis[indices] = color
            colored_pcd.colors = o3d.utility.Vector3dVector(colors_vis)
            o3d.visualization.draw_geometries(
                [colored_pcd], window_name="1. Plane Segmentation Result"
            )

        # =================================================================
        # 2. GROUPING BY PARALLELISM
        # =================================================================
        parallel_groups = self._group_parallel_planes(
            plane_normals, cfg.plane_angle_thresh
        )

        if cfg.show_planes_parallel_clustering and not cfg.no_image:
            colored_pcd_groups = copy.deepcopy(pcd_target)
            group_colors = [
                [random.random(), random.random(), random.random()]
                for _ in parallel_groups
            ]
            colors_groups = np.ones((len(pcd_target.points), 3)) * [0.5, 0.5, 0.5]
            for group_idx, group in enumerate(parallel_groups):
                for plane_idx in group:
                    colors_groups[plane_indices_list[plane_idx]] = group_colors[
                        group_idx
                    ]
            colored_pcd_groups.colors = o3d.utility.Vector3dVector(colors_groups)
            o3d.visualization.draw_geometries(
                [colored_pcd_groups], window_name="2. Parallel Clustering"
            )

        # Create pairs from Groups
        paired_planes: List[Tuple[int, int]] = []
        for group in parallel_groups:
            n = len(group)
            for i in range(n):
                for j in range(i + 1, n):
                    paired_planes.append((group[i], group[j]))

        # =================================================================
        # 3. EVALUATION OF EACH PAIR
        # =================================================================
        for count, (mmm, nnn) in enumerate(paired_planes):
            print(
                f"\n-------- Evaluating Pair: {count + 1}/{len(paired_planes)} --------"
            )

            if cfg.show_plane_pairs and not cfg.no_image:
                pair_colors = np.ones((len(pcd_target.points), 3)) * [0.6, 0.6, 0.6]
                pair_col = [random.random(), random.random(), random.random()]
                pair_colors[plane_indices_list[mmm]] = pair_col
                pair_colors[plane_indices_list[nnn]] = pair_col
                paired_pcd = copy.deepcopy(pcd_target)
                paired_pcd.colors = o3d.utility.Vector3dVector(pair_colors)
                o3d.visualization.draw_geometries(
                    [paired_pcd], window_name=f"3. Pair {count + 1} Highlight"
                )

            plane_i_points = np.asarray(
                pcd_target.select_by_index(plane_indices_list[mmm]).points
            )
            plane_j_points = np.asarray(
                pcd_target.select_by_index(plane_indices_list[nnn]).points
            )
            center_i = np.mean(plane_i_points, axis=0)
            center_j = np.mean(plane_j_points, axis=0)

            # Kinematic Constraint
            dist_plane = abs(np.dot(center_i - center_j, plane_normals[mmm]))
            print(
                f"Distance between planes: {dist_plane:.4f}m | Max Opening: {(g_cfg.f_pg - 2 * g_cfg.w_pg):.4f}m"
            )

            if dist_plane < g_cfg.g_pg or dist_plane > (g_cfg.f_pg - 2 * g_cfg.w_pg):
                print("Ignored: Part too thin or too thick for the gripper.")
                continue

            center_ij = (center_i + center_j) / 2
            dist_dir_i = (
                -1.0 if np.dot(center_ij - center_i, plane_normals[mmm]) > 0 else 1.0
            )
            dist_dir_j = (
                -1.0 if np.dot(center_ij - center_j, plane_normals[nnn]) > 0 else 1.0
            )

            dist_i = abs(np.dot((center_ij - center_i), plane_normals[mmm]))
            dist_j = abs(np.dot((center_ij - center_j), plane_normals[nnn]))

            projected_i_points = plane_i_points - dist_dir_i * np.outer(
                dist_i, plane_normals[mmm]
            )
            projected_j_points = plane_j_points - dist_dir_j * np.outer(
                dist_j, plane_normals[nnn]
            )

            pcd_proj_i = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(projected_i_points)
            )
            pcd_proj_j = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(projected_j_points)
            )

            pcd_orig_i = pcd_target.select_by_index(plane_indices_list[mmm])
            pcd_orig_j = pcd_target.select_by_index(plane_indices_list[nnn])

            if cfg.show_plane_pair_and_proj_in_pcd and not cfg.no_image:
                o3d.visualization.draw_geometries(
                    [pcd_orig_i, pcd_orig_j, pcd_proj_i, pcd_proj_j],
                    window_name="4. Center Projections",
                )

            # --- P1 Layer (Overlap) ---
            overlap_pcd_unfilter = self._extract_overlap_region(pcd_proj_i, pcd_proj_j)
            if overlap_pcd_unfilter is None:
                continue

            overlap_pcd, _ = overlap_pcd_unfilter.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=1.0
            )
            if cfg.show_proj_pts_p1 and not cfg.no_image:
                o3d.visualization.draw_geometries(
                    [overlap_pcd.translate([0, 0, 0.00001])],
                    window_name="5. P1 Layer (Overlap)",
                )

            # --- P2 Layer (Object Body) ---
            pts_btwn_p2, pts_beside = self._select_points_between_planes(
                pcd_target,
                center_i,
                center_j,
                plane_normals[mmm],
                cfg.margin_points_between_planes,
            )
            proj_p2 = self._project_points_to_plane(
                pts_btwn_p2, center_ij, plane_normals[mmm]
            )
            proj_pcd_p2, _ = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(proj_p2)
            ).remove_statistical_outlier(20, 1.0)
            if cfg.show_proj_pts_p2 and not cfg.no_image:
                o3d.visualization.draw_geometries(
                    [proj_pcd_p2], window_name="6. P2 Layer"
                )

            # --- P3 Layer (Finger Clearance) ---
            c_i_p3 = (
                center_i
                + (g_cfg.a_pg + g_cfg.w_pg + g_cfg.v_pg)
                * plane_normals[mmm]
                * dist_dir_i
            )
            c_j_p3 = (
                center_j
                + (g_cfg.a_pg + g_cfg.w_pg + g_cfg.v_pg)
                * plane_normals[nnn]
                * dist_dir_j
            )
            p3_i, pts_beside = self._select_points_between_planes(
                pts_beside,
                center_i,
                c_i_p3,
                plane_normals[mmm],
                cfg.margin_points_between_planes,
            )
            p3_j, pts_beside = self._select_points_between_planes(
                pts_beside,
                center_j,
                c_j_p3,
                plane_normals[nnn],
                cfg.margin_points_between_planes,
            )
            proj_p3 = self._project_points_to_plane(
                np.vstack((p3_i, p3_j)), center_ij, plane_normals[mmm]
            )
            proj_pcd_p3, _ = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(proj_p3)
            ).remove_statistical_outlier(50, 2.0)

            # --- P4 Layer (Base Clearance) ---
            c_i_p4 = center_ij + (y_pg / 2) * plane_normals[mmm] * dist_dir_i
            c_j_p4 = center_ij + (y_pg / 2) * plane_normals[nnn] * dist_dir_j
            p4_i, pts_beside = self._select_points_between_planes(
                pts_beside,
                c_i_p3,
                c_i_p4,
                plane_normals[mmm],
                cfg.margin_points_between_planes,
            )
            p4_j, pts_beside = self._select_points_between_planes(
                pts_beside,
                c_j_p3,
                c_j_p4,
                plane_normals[nnn],
                cfg.margin_points_between_planes,
            )
            proj_p4 = self._project_points_to_plane(
                np.vstack((p4_i, p4_j)), center_ij, plane_normals[mmm]
            )
            proj_pcd_p4, _ = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(proj_p4)
            ).remove_statistical_outlier(50, 3.0)

            # --- P5 Layer (Arm Clearance) ---
            c_i_p5 = center_ij + ((rd + g_cfg.rj) / 2) * plane_normals[mmm] * dist_dir_i
            c_j_p5 = center_ij + ((rd + g_cfg.rj) / 2) * plane_normals[nnn] * dist_dir_j
            p5_i, _ = self._select_points_between_planes(
                pts_beside,
                c_i_p4,
                c_i_p5,
                plane_normals[mmm],
                cfg.margin_points_between_planes,
            )
            p5_j, _ = self._select_points_between_planes(
                pts_beside,
                c_j_p4,
                c_j_p5,
                plane_normals[nnn],
                cfg.margin_points_between_planes,
            )
            proj_p5 = self._project_points_to_plane(
                np.vstack((p5_i, p5_j)), center_ij, plane_normals[mmm]
            )
            proj_pcd_p5 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj_p5))

            # --- OpenCV Contour Construction ---
            pca = PCA(n_components=3)
            pca.fit(np.asarray(pcd_target.points))
            dir1, dir2, center_pca = pca.components_[0], pca.components_[1], pca.mean_

            poly_p1, _, _ = self._get_plane_contour_polygon(
                overlap_pcd, dir1, dir2, center_pca, is_p2=False
            )
            poly_p2, segments_2d_p2, normals_2d_p2 = self._get_plane_contour_polygon(
                proj_pcd_p2, dir1, dir2, center_pca, is_p2=True
            )
            poly_p3, _, _ = self._get_plane_contour_polygon(
                proj_pcd_p3, dir1, dir2, center_pca, is_p2=False
            )
            poly_p4, _, _ = self._get_plane_contour_polygon(
                proj_pcd_p4, dir1, dir2, center_pca, is_p2=False
            )
            poly_p5, _, _ = self._get_plane_contour_polygon(
                proj_pcd_p5, dir1, dir2, center_pca, is_p2=False
            )

            # --- Geometry Cleanup ---
            poly_lists = [
                [self._clean_geom(p) for p in poly_p1],
                [self._clean_geom(p) for p in poly_p2],
                [self._clean_geom(p) for p in poly_p3],
                [self._clean_geom(p) for p in poly_p4],
                [self._clean_geom(p) for p in poly_p5],
            ]

            # --- TCP Grid Generation and Evaluation ---
            tcp_box, test_grid_points = self._generate_grid_by_spacing(
                segments_2d_p2,
                normals_2d_p2,
                depth=g_cfg.b_pg + g_cfg.c_pg,
                spacing_edge=g_cfg.z_pg / 5,
                spacing_normal=g_cfg.b_pg / 5,
            )

            points_and_gripper_boxes = self._create_gripper_bounding_box(
                test_grid_points, segments_2d_p2
            )

            min_area = 0.15 * (g_cfg.z_pg - 2 * g_cfg.rj) * (g_cfg.b_pg - 2 * g_cfg.rj)

            for edge_idx, segment_shapes in enumerate(points_and_gripper_boxes):
                pt1, pt2 = segments_2d_p2[edge_idx]
                seg_dir = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
                n_2d = np.array([-seg_dir[1], seg_dir[0]])

                # Orientation based on the original 2D reference frame
                x_axis = seg_dir[0] * dir1 + seg_dir[1] * dir2
                x_axis = x_axis / np.linalg.norm(x_axis)
                z_axis = n_2d[0] * dir1 + n_2d[1] * dir2
                z_axis = z_axis / np.linalg.norm(z_axis)
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)

                R = np.column_stack((x_axis, y_axis, z_axis))

                for shape in segment_shapes:
                    pt = shape["point"]
                    r = shape["rectangles"]

                    rect1_geom, rect2_geom = Polygon(r[0]), Polygon(r[1])
                    rect3_geom, rect4_geom = Polygon(r[2]), Polygon(r[3])
                    rect5_geom = Polygon(r[4])

                    area = sum(p.intersection(rect5_geom).area for p in poly_lists[0])
                    if area <= min_area:
                        continue
                    if any(
                        p.intersects(rect3_geom) or p.intersects(rect4_geom)
                        for p in poly_lists[1]
                    ):
                        continue
                    if any(
                        p.intersects(rect1_geom) or p.intersects(rect2_geom)
                        for p in poly_lists[2]
                    ):
                        continue
                    if any(
                        p.intersects(rect3_geom) or p.intersects(rect4_geom)
                        for p in poly_lists[3]
                    ):
                        continue
                    if any(p.intersects(rect4_geom) for p in poly_lists[4]):
                        continue

                    # Success! Register the candidate
                    pt_3d = center_pca + pt[0] * dir1 + pt[1] * dir2
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :3] = R
                    pose_4x4[:3, 3] = pt_3d

                    max_a = max(
                        (g_cfg.z_pg - 2 * g_cfg.rj) * (g_cfg.b_pg - 2 * g_cfg.rj), 1e-9
                    )
                    s_area = np.clip((area - 0.15 * max_a) / (0.85 * max_a), 0.0, 1.0)
                    dist_center = np.linalg.norm(
                        pt_3d - np.mean(original_points, axis=0)
                    )
                    s_center = np.clip(1.0 - (dist_center / 1.0), 0.0, 1.0)

                    final_score = 0.1 * s_center + 0.9 * s_area

                    if final_score >= cfg.min_score:
                        cand = GraspCandidate(
                            transform=pose_4x4,
                            score=final_score,
                            contact_point=pt_3d,
                            approach_vector=-pose_4x4[:3, 2],
                            score_details={
                                "area_score": s_area,
                                "center_score": s_center,
                                "total_area": area,
                            },
                        )
                        self.candidates.append(cand)

        self.valid_candidates = sorted(
            self.candidates, key=lambda x: x.score, reverse=True
        )
        print(
            f"\n🏆 [ParallelSampler] Completed! Total valid grasps: {len(self.valid_candidates)}"
        )
        return self.valid_candidates

    # =========================================================================
    # MATHEMATICAL AND GEOMETRIC AUXILIARY FUNCTIONS
    # =========================================================================
    @staticmethod
    def _normalize_plane(
        a: float, b: float, c: float, d: float
    ) -> Tuple[np.ndarray, float]:
        """Ensures the plane normal has unit length."""
        n = np.array([a, b, c], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("Invalid plane normal (zero length).")
        return n / norm, d / norm

    @staticmethod
    def _angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
        """Calculates the angle in degrees between two normal vectors."""
        cosv = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return np.degrees(np.arccos(cosv))

    @staticmethod
    def _refit_plane_from_points(
        points_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fits a new mathematical plane to a set of 3D points using SVD."""
        P = np.asarray(points_xyz, dtype=float)
        if len(P) < 3:
            raise ValueError(
                "At least 3 points are required to fit a plane."
            )
        centroid = P.mean(axis=0)
        Q = P - centroid
        _, _, vt = np.linalg.svd(Q, full_matrices=False)
        normal = vt[-1, :]
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        return np.array([normal[0], normal[1], normal[2], d], dtype=float), normal

    def _merge_coplanar_planes(
        self,
        p_models: List[np.ndarray],
        p_normals: List[np.ndarray],
        p_indices: List[np.ndarray],
        all_pts: np.ndarray,
        angle_thresh_deg: float,
        offset_thresh: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Merges multiple small RANSAC planes into larger continuous surfaces."""
        planes: List[Dict[str, Any]] = []
        for m, n_unit, idxs in zip(p_models, p_normals, p_indices):
            a, b, c, d = m
            n_hat, d_hat = self._normalize_plane(a, b, c, d)
            if np.dot(n_hat, n_unit) < 0:
                n_hat, d_hat = -n_hat, -d_hat
            planes.append({"n": n_hat, "d": d_hat, "idxs": np.asarray(idxs, dtype=int)})

        changed = True
        while changed:
            changed = False
            N = len(planes)
            if N <= 1:
                break
            merged_pair = None

            for i in range(N):
                for j in range(i + 1, N):
                    n1, d1 = planes[i]["n"], planes[i]["d"]
                    n2, d2 = planes[j]["n"], planes[j]["d"]
                    n2_cmp, d2_cmp = (-n2, -d2) if np.dot(n1, n2) < 0 else (n2, d2)

                    if (
                        self._angle_deg(n1, n2_cmp) <= angle_thresh_deg
                        and abs(d1 - d2_cmp) <= offset_thresh
                    ):
                        merged_pair = (i, j)
                        break
                if merged_pair is not None:
                    break

            if merged_pair is not None:
                i, j = merged_pair
                idxs_merged = np.unique(
                    np.concatenate([planes[i]["idxs"], planes[j]["idxs"]], axis=0)
                )
                pts = all_pts[idxs_merged]
                model_new, n_new = self._refit_plane_from_points(pts)
                n_hat, d_hat = self._normalize_plane(*model_new)

                if np.dot(n_hat, planes[i]["n"]) < 0:
                    n_hat, d_hat = -n_hat, -d_hat
                planes[i] = {"n": n_hat, "d": d_hat, "idxs": idxs_merged}
                del planes[j]
                changed = True

        res_models = [
            np.array([pl["n"][0], pl["n"][1], pl["n"][2], pl["d"]], dtype=float)
            for pl in planes
        ]
        res_normals = [pl["n"] for pl in planes]
        res_indices = [pl["idxs"] for pl in planes]
        return res_models, res_normals, res_indices

    def _group_parallel_planes(
        self, plane_normals: List[np.ndarray], angle_thresh_deg: float
    ) -> List[List[int]]:
        """
        Groups indices of planes that are parallel to each other.
        Order matters: The reference plane should always be the largest (index 0 of the list).
        """
        unclustered = list(range(len(plane_normals)))
        parallel_groups: List[List[int]] = []

        while len(unclustered) > 0:
            ref_idx = unclustered[0]
            current_group = [ref_idx]
            unclustered.pop(0)

            ref_normal = plane_normals[ref_idx]
            to_remove = []

            for other in unclustered:
                cos_theta = np.clip(np.dot(ref_normal, plane_normals[other]), -1.0, 1.0)
                if degrees(acos(abs(cos_theta))) <= angle_thresh_deg:
                    current_group.append(other)
                    to_remove.append(other)

            for i in to_remove:
                unclustered.remove(i)

            parallel_groups.append(current_group)

        return parallel_groups

    def _extract_overlap_region(
        self, proj_A: o3d.geometry.PointCloud, proj_B: o3d.geometry.PointCloud
    ) -> Optional[o3d.geometry.PointCloud]:
        """Extracts the 3D region where two projected planes overlap."""
        if len(proj_A.points) == 0 or len(proj_B.points) == 0:
            return None

        dA = np.asarray(proj_A.compute_nearest_neighbor_distance())
        dB = np.asarray(proj_B.compute_nearest_neighbor_distance())
        med_A = np.median(dA) if len(dA) > 0 else 0
        med_B = np.median(dB) if len(dB) > 0 else 0

        th = 1.2 * max(med_A, med_B)
        if th == 0:
            th = 0.001

        kdtree_A = o3d.geometry.KDTreeFlann(proj_A)
        matched_B = []
        for p in np.asarray(proj_B.points):
            k, _, _ = kdtree_A.search_radius_vector_3d(p, th)
            if k > 0:
                matched_B.append(p)

        if not matched_B:
            return None
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(matched_B)))

    def _select_points_between_planes(
        self,
        pcd_pts: Any,
        center_a: np.ndarray,
        center_b: np.ndarray,
        normal: np.ndarray,
        margin: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filters points located physically between two planar slices."""
        points = (
            np.asarray(pcd_pts.points)
            if isinstance(pcd_pts, o3d.geometry.PointCloud)
            else pcd_pts
        )
        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        d_a = np.dot(points - center_a, normal)
        d_b = np.dot(points - center_b, normal)
        mask = (d_a * d_b <= 0) | (np.abs(d_a) <= margin) | (np.abs(d_b) <= margin)

        return points[mask], points[~mask]

    def _project_points_to_plane(
        self, points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
    ) -> np.ndarray:
        """Projects a 3D point array against a mathematical planar surface."""
        if len(points) == 0:
            return np.empty((0, 3))
        v = points - plane_point
        d = np.dot(v, plane_normal)
        return points - np.outer(d, plane_normal)

    def _get_plane_contour_polygon(
        self,
        p_cloud: o3d.geometry.PointCloud,
        d1: np.ndarray,
        d2: np.ndarray,
        ctr: np.ndarray,
        is_p2: bool = False,
    ) -> Tuple[List[Polygon], List[np.ndarray], List[np.ndarray]]:
        """Generates 2D polygons (Shapely) from projected point clouds using OpenCV."""
        if p_cloud.is_empty() or len(p_cloud.points) <= 50:
            return [Polygon()], [], []

        pts = np.asarray(p_cloud.points)
        pts_2d = np.dot(pts - ctr, np.vstack([d1, d2]).T)
        min_pt, max_pt = pts_2d.min(axis=0), pts_2d.max(axis=0)

        ranges = max_pt - min_pt
        if np.max(ranges) == 0:
            return [Polygon()], [], []

        scale = 512.0 / np.max(ranges)
        pad = self.config.contour_image_padding

        pts_img = np.int32((pts_2d - min_pt) * scale) + pad
        img_size = ((max_pt - min_pt) * scale).astype(int) + 2 * pad

        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for p in pts_img:
            cv2.circle(img, tuple(p), 1, 255, -1)

        px_gap = 3
        k, k_open = max(3, int(round(px_gap * 2))), max(3, int(round(px_gap * 0.8)))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        ff = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(ff, None, (0, 0), 255)
        ff = ff[1:-1, 1:-1]
        filled = cv2.bitwise_or(
            mask, cv2.bitwise_not(cv2.bitwise_not(ff) & cv2.bitwise_not(mask))
        )

        num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        clean = np.zeros_like(filled)
        for ix in range(1, num):
            if stats[ix, cv2.CC_STAT_AREA] >= (k * k) * 2:
                clean[labels == ix] = 255

        contours, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polys, segments_2d, normals_2d = [], [], []
        for cnt in contours:
            eps = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            pts_2d_back = (approx.astype(np.float32) - pad) / scale + min_pt
            if len(pts_2d_back) >= 3:
                polys.append(Polygon(pts_2d_back))

            if is_p2:
                for ix in range(len(pts_2d_back)):
                    pt1_2d = pts_2d_back[ix]
                    pt2_2d = pts_2d_back[(ix + 1) % len(pts_2d_back)]
                    vec = pt2_2d - pt1_2d
                    L = np.linalg.norm(vec)
                    if L == 0:
                        continue
                    dir_u = vec / L
                    segments_2d.append([pt1_2d, pt2_2d])
                    normals_2d.append(np.array([-dir_u[1], dir_u[0]]))

        return (polys if polys else [Polygon()]), segments_2d, normals_2d

    def _generate_grid_by_spacing(
        self,
        segments_2d: List[List[np.ndarray]],
        normals_2d: List[np.ndarray],
        depth: float,
        spacing_edge: float,
        spacing_normal: float,
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """Generates TCP candidate points based on the outer contour and finger depth."""
        rectangles, all_grid_points = [], []
        eps = 1e-9
        for (pt1, pt2), n in zip(segments_2d, normals_2d):
            pt1, pt2, n = np.array(pt1), np.array(pt2), np.array(n) / np.linalg.norm(n)
            vec = pt2 - pt1
            seg_len = np.linalg.norm(vec)
            dir_unit = vec / seg_len
            num_w = int(np.floor((seg_len - eps) / spacing_edge) + 1)
            start_w = (seg_len - (num_w - 1) * spacing_edge) / 2.0
            num_d = int(np.floor((depth - eps) / spacing_normal) + 1)
            start_d = (depth - (num_d - 1) * spacing_normal) / 2.0

            if num_w < 1 or num_d < 1:
                continue

            offset = -n * depth
            rectangles.append([pt1 + offset, pt2 + offset, pt2, pt1])

            grid_pts = []
            for iw in range(num_w):
                for jw in range(num_d):
                    pt = (
                        (pt1 + offset)
                        + dir_unit * (iw * spacing_edge + start_w)
                        + n * (jw * spacing_normal + start_d)
                    )
                    grid_pts.append(pt)
            all_grid_points.append(np.array(grid_pts))

        return rectangles, all_grid_points

    def _create_gripper_bounding_box(
        self, grid_points: List[np.ndarray], segments_2d: List[List[np.ndarray]]
    ) -> List[List[Dict[str, Any]]]:
        """Creates the 6 parametric rectangular boxes representing the robot arm in 2D."""
        all_shapes = []
        g_cfg = self.gripper.config

        for pts, (pt1, pt2) in zip(grid_points, segments_2d):
            seg_dir = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
            normal = np.array([-seg_dir[1], seg_dir[0]])
            mid = (pt1 + pt2) / 2

            segment_shapes = []
            for pt in pts:
                grid_edge_distance = np.dot(mid - pt, normal)
                rects = []

                # 1. P1: Finger front safe space
                c1 = pt - normal * (g_cfg.x_pg + g_cfg.rj)
                hw = (g_cfg.e_pg + 2 * (g_cfg.i_pg + g_cfg.rj)) / 2
                rects.append(
                    [
                        c1 + seg_dir * hw,
                        c1 + seg_dir * hw + normal * (g_cfg.x_pg + g_cfg.rj),
                        c1 - seg_dir * hw + normal * (g_cfg.x_pg + g_cfg.rj),
                        c1 - seg_dir * hw,
                    ]
                )

                # 2. P2: Finger length and touch zone
                c2 = pt
                rects.append(
                    [
                        c2 + seg_dir * hw,
                        c2
                        + seg_dir * hw
                        + normal * (g_cfg.b_pg + g_cfg.c_pg + g_cfg.rj),
                        c2
                        - seg_dir * hw
                        + normal * (g_cfg.b_pg + g_cfg.c_pg + g_cfg.rj),
                        c2 - seg_dir * hw,
                    ]
                )

                # 3. P3: Gripper lower base
                c3 = c2 + normal * (g_cfg.b_pg + g_cfg.c_pg + g_cfg.rj)
                hb = (
                    max(
                        g_cfg.l_pg + 2 * g_cfg.m_pg,
                        g_cfg.o_pg + 2 * g_cfg.p_pg,
                        g_cfg.e_pg + 2 * g_cfg.i_pg,
                    )
                    + 2 * g_cfg.rj
                ) / 2
                hth = g_cfg.d_pg + g_cfg.t_pg + g_cfg.u_pg + g_cfg.rj
                rects.append(
                    [
                        c3 + seg_dir * hb,
                        c3 + seg_dir * hb + normal * hth,
                        c3 - seg_dir * hb + normal * hth,
                        c3 - seg_dir * hb,
                    ]
                )

                # 4. P4: Robot upper base and main arm
                c4 = c3 + normal * hth
                ha = (max(g_cfg.ra, g_cfg.rb) + g_cfg.re + 2 * g_cfg.rj) / 2
                hta = g_cfg.rc + g_cfg.rf + 2 * g_cfg.rj
                rects.append(
                    [
                        c4 + seg_dir * ha,
                        c4 - seg_dir * ha,
                        c4 - seg_dir * ha + normal * hta,
                        c4 + seg_dir * ha + normal * hta,
                    ]
                )

                # 5. P5: Gripper internal free area
                c5 = pt
                harea = (g_cfg.z_pg - 2 * g_cfg.rj) / 2
                htarea = g_cfg.b_pg - 2 * g_cfg.rj
                rects.append(
                    [
                        c5 + seg_dir * harea,
                        c5 + seg_dir * harea + normal * htarea,
                        c5 - seg_dir * harea + normal * htarea,
                        c5 - seg_dir * harea,
                    ]
                )

                # 6. P6: Safety Back Space
                c6 = c4 + normal * hta
                htb = grid_edge_distance + g_cfg.x_pg + g_cfg.rj
                rects.append(
                    [
                        c6 + seg_dir * ha,
                        c6 + seg_dir * ha + normal * htb,
                        c6 - seg_dir * ha + normal * htb,
                        c6 - seg_dir * ha,
                    ]
                )

                segment_shapes.append({"point": pt, "rectangles": rects})
            all_shapes.append(segment_shapes)
        return all_shapes

    def _clean_geom(self, geom: Any) -> Any:
        """Repairs and simplifies invalid polygons (e.g., self-intersections) for Shapely."""
        if geom.is_empty:
            return geom
        g = geom
        if not g.is_valid:
            g = make_valid(g)
        if not g.is_valid:
            g = g.buffer(0)
        try:
            g = set_precision(g, 1e-9)
        except:
            pass
        try:
            if hasattr(g, "geoms"):
                g = unary_union(g)
        except:
            pass
        return g
