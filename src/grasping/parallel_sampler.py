import copy
import numpy as np
import open3d as o3d
import cv2
import itertools
from sklearn.decomposition import PCA
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid
from shapely.prepared import prep
from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Optional

from src.grasping.base_sampler import BaseGraspSampler, GraspCandidate
from src.grippers.parallel_gripper import ParallelGripper


class ParallelScoreDetails(TypedDict):
    area_score: float
    center_score: float
    total_area: float


@dataclass
class ParallelSamplerConfig:
    plane_angle_thresh: float = 5.0
    min_score: float = 0.3
    margin_points_between_planes: float = 0.0015
    grid_spacing_edge: float = 0.004
    grid_spacing_normal: float = 0.006

    # Translation of Hanyu's parameters to Meters
    ransac_distance_thresh: float = 0.001  # 1mm in Hanyu's script
    merge_offset_thresh: float = 0.002  # 2mm in Hanyu's script


class ParallelGraspSampler(
    BaseGraspSampler[ParallelGripper, ParallelSamplerConfig, ParallelScoreDetails]
):
    """
    Sampler for Parallel Grippers.
    Strict 1:1 functional replica of the original main_script.py logic.
    """

    def __init__(
        self, gripper: ParallelGripper, config: Optional[ParallelSamplerConfig] = None
    ):
        if config is None:
            config = ParallelSamplerConfig()
        super().__init__(gripper, config)

    def sample_grasps(self, pcd: o3d.geometry.PointCloud) -> List[GraspCandidate]:
        self.clear_candidates()
        print("[ParallelSampler] Step 1: Extracting, merging and pairing planes...")

        # 1. Plane Segmentation & Merging (Exactly like main_script.py)
        plane_models, plane_normals, plane_indices = self._extract_and_merge_planes(pcd)
        if not plane_normals:
            print("[ParallelSampler] No planes found.")
            return []

        # 2. Parallel Grouping and Pairing (Exactly like main_script.py)
        paired_planes = self._group_and_pair_planes(plane_normals)
        print(f"[ParallelSampler] Found {len(paired_planes)} potential plane pairs.")

        # 3. Evaluate each pair
        for idx, (mmm, nnn) in enumerate(paired_planes):
            print(f"  -> Processing pair {idx + 1}/{len(paired_planes)}...")

            valid, center_ij, dir1, dir2, center_pca, poly_lists, contour_data = (
                self._analyze_plane_pair(pcd, mmm, nnn, plane_normals, plane_indices)
            )
            if not valid:
                continue

            segments_2d, normals_2d = contour_data
            plane_normal_3d = plane_normals[mmm]

            print("     Generating TCP grid...")
            poses_3d, shapely_boxes = self._generate_tcp_candidates(
                segments_2d, normals_2d, center_pca, dir1, dir2, plane_normal_3d
            )

            print(f"     Testing {len(poses_3d)} poses for collisions...")
            valid_poses, valid_areas = self._evaluate_collisions_2d(
                poses_3d, shapely_boxes, poly_lists
            )

            if valid_poses:
                print(f"     ✅ {len(valid_poses)} poses survived collision checks.")
                scores = self._calculate_scores(valid_poses, valid_areas, pcd)

                for pose, score_tuple in zip(valid_poses, scores):
                    final_score, area_score, center_score, area = score_tuple

                    if final_score >= self.config.min_score:
                        cand = GraspCandidate(
                            transform=pose,
                            score=final_score,
                            contact_point=pose[:3, 3],
                            approach_vector=-pose[:3, 2],  # Point INTO the object
                            score_details={
                                "area_score": area_score,
                                "center_score": center_score,
                                "total_area": area,
                            },
                        )
                        self.candidates.append(cand)

        self.valid_candidates = sorted(
            self.candidates, key=lambda x: x.score, reverse=True
        )
        print(
            f"\n🏆 [ParallelSampler] Finished! Total valid grasps: {len(self.valid_candidates)}"
        )
        return self.valid_candidates

    # =========================================================================
    # Phase 1: Plane Extraction, Merging & Pairing (Hanyu's Exact Math)
    # =========================================================================

    @staticmethod
    def _normalize_plane(a, b, c, d):
        n = np.array([a, b, c], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("Invalid plane normal with zero length.")
        return n / norm, d / norm

    @staticmethod
    def _angle_deg(n1, n2):
        cosv = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return np.degrees(np.arccos(cosv))

    @staticmethod
    def _refit_plane_from_points(points_xyz):
        P = np.asarray(points_xyz, dtype=float)
        if len(P) < 3:
            raise ValueError("Need at least 3 points to fit a plane.")
        centroid = P.mean(axis=0)
        Q = P - centroid
        _, _, vt = np.linalg.svd(Q, full_matrices=False)
        normal = vt[-1, :]
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        return np.array([normal[0], normal[1], normal[2], d], dtype=float), normal

    def _extract_and_merge_planes(self, pcd: o3d.geometry.PointCloud):
        """Extracts planes with RANSAC and merges them using Hanyu's custom logic."""
        working_pcd = copy.deepcopy(pcd)
        original_points = np.asarray(pcd.points)
        original_indices = np.arange(len(original_points))

        plane_indices_list = []
        plane_models = []
        plane_normals = []

        # 1. RANSAC Extraction
        for _ in range(200):  # max_planes
            if len(working_pcd.points) < 50:
                break
            try:
                plane_model, inliers = working_pcd.segment_plane(
                    distance_threshold=self.config.ransac_distance_thresh,
                    ransac_n=3,
                    num_iterations=1000,
                )
            except RuntimeError:
                break

            if len(inliers) < 50:
                break

            original_idx = original_indices[inliers]
            plane_indices_list.append(original_idx)
            plane_models.append(plane_model)

            n = np.asarray(plane_model[0:3])
            plane_normals.append(n / np.linalg.norm(n))

            working_pcd = working_pcd.select_by_index(inliers, invert=True)
            original_indices = np.delete(original_indices, inliers)

        # 2. Hanyu's Merging Logic
        planes = []
        for model, n_unit, idxs in zip(plane_models, plane_normals, plane_indices_list):
            a, b, c, d = model
            n_hat, d_hat = self._normalize_plane(a, b, c, d)
            if np.dot(n_hat, n_unit) < 0:
                n_hat = -n_hat
                d_hat = -d_hat
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

                    if np.dot(n1, n2) < 0:
                        n2_cmp = -n2
                        d2_cmp = -d2
                    else:
                        n2_cmp, d2_cmp = n2, d2

                    angle = self._angle_deg(n1, n2_cmp)
                    offset_diff = abs(d1 - d2_cmp)

                    if (
                        angle <= self.config.plane_angle_thresh
                        and offset_diff <= self.config.merge_offset_thresh
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
                pts = original_points[idxs_merged]
                model_new, n_new = self._refit_plane_from_points(pts)
                n_hat, d_hat = self._normalize_plane(*model_new)

                if np.dot(n_hat, planes[i]["n"]) < 0:
                    n_hat = -n_hat
                    d_hat = -d_hat
                planes[i] = {"n": n_hat, "d": d_hat, "idxs": idxs_merged}
                del planes[j]
                changed = True

        new_plane_models = [
            np.array([pl["n"][0], pl["n"][1], pl["n"][2], pl["d"]], dtype=float)
            for pl in planes
        ]
        new_normals = [pl["n"] for pl in planes]
        new_indices_list = [pl["idxs"] for pl in planes]

        return new_plane_models, new_normals, new_indices_list

    def _group_and_pair_planes(self, plane_normals):
        """Replicates the parallel_groups logic from main_script.py"""
        unclustered = set(range(len(plane_normals)))
        parallel_groups = []

        while unclustered:
            idx = unclustered.pop()
            ref_normal = plane_normals[idx]
            current_group = [idx]
            to_remove = []

            for other in unclustered:
                cos_theta = np.clip(np.dot(ref_normal, plane_normals[other]), -1.0, 1.0)
                angle = np.degrees(np.arccos(abs(cos_theta)))
                if angle <= self.config.plane_angle_thresh:
                    current_group.append(other)
                    to_remove.append(other)

            for i in to_remove:
                unclustered.remove(i)
            parallel_groups.append(current_group)

        paired_planes = []
        for group in parallel_groups:
            n = len(group)
            for i in range(n):
                for j in range(i + 1, n):
                    paired_planes.append((group[i], group[j]))

        return paired_planes

    # =========================================================================
    # Phase 2: Layer Extraction & OpenCV
    # =========================================================================
    def _analyze_plane_pair(self, pcd, mmm, nnn, plane_normals, plane_indices_list):
        c = self.gripper.config

        plane_i_points = np.asarray(pcd.select_by_index(plane_indices_list[mmm]).points)
        plane_j_points = np.asarray(pcd.select_by_index(plane_indices_list[nnn]).points)
        center_i = np.mean(plane_i_points, axis=0)
        center_j = np.mean(plane_j_points, axis=0)

        normal_i = plane_normals[mmm]
        normal_j = plane_normals[nnn]

        dist_plane = abs(np.dot(center_i - center_j, normal_i))
        if dist_plane < c.g_pg or dist_plane > (c.f_pg - 2 * c.w_pg):
            return False, None, None, None, None, None, None

        center_ij = (center_i + center_j) / 2

        dist_dir_i = -1.0 if np.dot(center_ij - center_i, normal_i) > 0 else 1.0
        dist_dir_j = -1.0 if np.dot(center_ij - center_j, normal_j) > 0 else 1.0

        pca = PCA(n_components=3)
        pca.fit(np.asarray(pcd.points))
        dir1, dir2 = pca.components_[0], pca.components_[1]
        center_pca = pca.mean_

        margin = self.config.margin_points_between_planes

        # --- 1. Layer P1 (Overlap) ---
        dist_i = abs(np.dot((center_ij - center_i), normal_i))
        dist_j = abs(np.dot((center_ij - center_j), normal_j))

        proj_i = plane_i_points - dist_dir_i * np.outer(dist_i, normal_i)
        proj_j = plane_j_points - dist_dir_j * np.outer(dist_j, normal_j)

        pcd_proj_i = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj_i))
        pcd_proj_j = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj_j))
        overlap_pcd = self._extract_overlap_region(pcd_proj_i, pcd_proj_j)

        if overlap_pcd is None:
            return False, None, None, None, None, None, None

        # --- 2. Layer P2 (Between Planes) ---
        pts_between_p2, pts_beside = self._select_points_between_planes(
            pcd, center_i, center_j, normal_i, margin
        )
        proj_p2 = self._project_to_plane(pts_between_p2, center_ij, normal_i)
        pcd_p2, _ = proj_p2.remove_statistical_outlier(20, 1.0)

        # --- 3. Layer P3 (Finger Clearance) ---
        c_i_p3 = center_i + (c.a_pg + c.w_pg + c.v_pg) * normal_i * dist_dir_i
        c_j_p3 = center_j + (c.a_pg + c.w_pg + c.v_pg) * normal_j * dist_dir_j
        pts_p3_i, pts_beside = self._select_points_between_planes(
            pts_beside, center_i, c_i_p3, normal_i, margin
        )
        pts_p3_j, pts_beside = self._select_points_between_planes(
            pts_beside, center_j, c_j_p3, normal_j, margin
        )
        proj_p3 = self._project_to_plane(
            np.vstack((pts_p3_i, pts_p3_j)), center_ij, normal_i
        )
        pcd_p3, _ = proj_p3.remove_statistical_outlier(50, 2.0)

        # --- 4. Layer P4 (Base Clearance) ---
        y_pg = max(
            c.q_pg + 2 * c.r_pg, c.h_pg + 2 * c.k_pg, c.f_pg + 2 * (c.a_pg + c.v_pg)
        )
        c_i_p4 = center_ij + (y_pg / 2) * normal_i * dist_dir_i
        c_j_p4 = center_ij + (y_pg / 2) * normal_j * dist_dir_j
        pts_p4_i, pts_beside = self._select_points_between_planes(
            pts_beside, c_i_p3, c_i_p4, normal_i, margin
        )
        pts_p4_j, pts_beside = self._select_points_between_planes(
            pts_beside, c_j_p3, c_j_p4, normal_j, margin
        )
        proj_p4 = self._project_to_plane(
            np.vstack((pts_p4_i, pts_p4_j)), center_ij, normal_i
        )
        pcd_p4, _ = proj_p4.remove_statistical_outlier(50, 3.0)

        # --- 5. Layer P5 (Arm Clearance) ---
        rd = max(c.ra, c.rb)
        c_i_p5 = center_ij + ((rd + c.rj) / 2) * normal_i * dist_dir_i
        c_j_p5 = center_ij + ((rd + c.rj) / 2) * normal_j * dist_dir_j
        pts_p5_i, _ = self._select_points_between_planes(
            pts_beside, c_i_p4, c_i_p5, normal_i, margin
        )
        pts_p5_j, _ = self._select_points_between_planes(
            pts_beside, c_j_p4, c_j_p5, normal_j, margin
        )
        proj_p5 = self._project_to_plane(
            np.vstack((pts_p5_i, pts_p5_j)), center_ij, normal_i
        )

        # Build 2D Polygons via OpenCV
        poly_p1 = self._get_plane_contour_polygon_pca(
            overlap_pcd, dir1, dir2, center_pca
        )
        poly_p2 = self._get_plane_contour_polygon_pca(pcd_p2, dir1, dir2, center_pca)
        poly_p3 = self._get_plane_contour_polygon_pca(pcd_p3, dir1, dir2, center_pca)
        poly_p4 = self._get_plane_contour_polygon_pca(pcd_p4, dir1, dir2, center_pca)
        poly_p5 = self._get_plane_contour_polygon_pca(proj_p5, dir1, dir2, center_pca)

        poly_lists = [
            [self._clean_geom(p) for p in poly_p1],
            [self._clean_geom(p) for p in poly_p2],
            [self._clean_geom(p) for p in poly_p3],
            [self._clean_geom(p) for p in poly_p4],
            [self._clean_geom(p) for p in poly_p5],
        ]

        segments, normals = self._extract_contour_segments(poly_p2)

        return True, center_ij, dir1, dir2, center_pca, poly_lists, (segments, normals)

    def _extract_overlap_region(self, proj_A, proj_B):
        if len(proj_A.points) == 0 or len(proj_B.points) == 0:
            return None

        dA = np.asarray(proj_A.compute_nearest_neighbor_distance())
        dB = np.asarray(proj_B.compute_nearest_neighbor_distance())
        threshold = 1.2 * max(
            np.median(dA) if len(dA) > 0 else 0, np.median(dB) if len(dB) > 0 else 0
        )
        if threshold == 0:
            threshold = 0.001

        kdtree_A = o3d.geometry.KDTreeFlann(proj_A)
        matched_B = []
        for p in np.asarray(proj_B.points):
            k, _, _ = kdtree_A.search_radius_vector_3d(p, threshold)
            if k > 0:
                matched_B.append(p)

        if not matched_B:
            return None
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(matched_B)))

    def _select_points_between_planes(self, pts, center_a, center_b, normal, margin):
        if isinstance(pts, o3d.geometry.PointCloud):
            pts = np.asarray(pts.points)
        if len(pts) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        d_a = np.dot(pts - center_a, normal)
        d_b = np.dot(pts - center_b, normal)
        mask = (d_a * d_b <= 0) | (np.abs(d_a) <= margin) | (np.abs(d_b) <= margin)
        return pts[mask], pts[~mask]

    def _project_to_plane(self, points, plane_point, plane_normal):
        if len(points) == 0:
            return o3d.geometry.PointCloud()
        v = points - plane_point
        d = np.dot(v, plane_normal)
        proj = points - np.outer(d, plane_normal)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj))

    def _get_plane_contour_polygon_pca(self, pcd, dir1, dir2, center):
        if pcd is None or pcd.is_empty():
            return [Polygon()]

        points = np.asarray(pcd.points)
        if len(points) < 3:
            return [Polygon()]

        pts_2d = np.dot(points - center, np.vstack([dir1, dir2]).T)
        min_pt = pts_2d.min(axis=0)
        max_pt = pts_2d.max(axis=0)
        ranges = max_pt - min_pt

        if np.max(ranges) == 0:
            return [Polygon()]

        scale = 512.0 / np.max(ranges)
        padding = 10
        pts_img = np.int32((pts_2d - min_pt) * scale) + padding
        img_size = ((max_pt - min_pt) * scale).astype(int) + 2 * padding

        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for pt in pts_img:
            cv2.circle(img, tuple(pt), 1, 255, -1)

        px_gap = 3
        k = max(3, int(round(px_gap * 2)))
        k_open = max(3, int(round(px_gap * 0.8)))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        ff = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(ff, None, (0, 0), 255)
        ff = ff[1:-1, 1:-1]
        holes = cv2.bitwise_not(ff) & cv2.bitwise_not(mask)
        filled = cv2.bitwise_or(mask, cv2.bitwise_not(holes))

        num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        min_area_px = (k * k) * 2
        clean = np.zeros_like(filled)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                clean[labels == i] = 255

        contours, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
            pts_2d_back = (approx.astype(np.float32) - padding) / scale + min_pt
            if len(pts_2d_back) >= 3:
                polygons.append(Polygon(pts_2d_back))

        return polygons if polygons else [Polygon()]

    def _extract_contour_segments(self, polys):
        segments = []
        normals = []
        for poly in polys:
            if poly.is_empty:
                continue
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])
                segments.append([p1, p2])
                vec = p2 - p1
                L = np.linalg.norm(vec)
                if L == 0:
                    normals.append(np.array([0, 0]))
                else:
                    dir_u = vec / L
                    normals.append(np.array([-dir_u[1], dir_u[0]]))
        return segments, normals

    def _clean_geom(self, geom):
        if geom.is_empty:
            return geom
        g = geom
        if not g.is_valid:
            g = make_valid(g)
        if not g.is_valid:
            g = g.buffer(0)
        return g

    # =========================================================================
    # Phase 3 & 4: Grid, Boxes, and Collision
    # =========================================================================
    def _generate_tcp_candidates(
        self, segments_2d, normals_2d, center_pca, dir1, dir2, plane_normal_3d
    ):
        c = self.gripper.config
        depth = c.b_pg + c.c_pg
        spacing_edge = self.config.grid_spacing_edge
        spacing_normal = self.config.grid_spacing_normal

        poses_3d = []
        shapely_boxes = []

        for (pt1, pt2), n_2d in zip(segments_2d, normals_2d):
            pt1, pt2 = np.array(pt1), np.array(pt2)
            n_2d = np.array(n_2d) / np.linalg.norm(n_2d)

            dir_vec = pt2 - pt1
            seg_len = np.linalg.norm(dir_vec)
            if seg_len == 0:
                continue
            seg_dir = dir_vec / seg_len

            num_w = int(np.floor((seg_len - 1e-9) / spacing_edge) + 1)
            start_w = (seg_len - (num_w - 1) * spacing_edge) / 2.0
            num_d = int(np.floor((depth - 1e-9) / spacing_normal) + 1)
            start_d = (depth - (num_d - 1) * spacing_normal) / 2.0

            offset = -n_2d * depth
            p1 = pt1 + offset
            mid_segment = (pt1 + pt2) / 2.0

            for i in range(num_w):
                for j in range(num_d):
                    alpha = i * spacing_edge + start_w
                    beta = j * spacing_normal + start_d
                    pt_2d = p1 + seg_dir * alpha + n_2d * beta

                    grid_edge_distance = np.dot(mid_segment - pt_2d, n_2d)

                    rects = self._create_2d_gripper_boxes(
                        pt_2d, seg_dir, n_2d, grid_edge_distance
                    )
                    shapely_boxes.append([Polygon(rect) for rect in rects])

                    pt_3d = center_pca + pt_2d[0] * dir1 + pt_2d[1] * dir2

                    z_axis = plane_normal_3d
                    x_axis = seg_dir[0] * dir1 + seg_dir[1] * dir2
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)

                    R = np.column_stack((x_axis, y_axis, z_axis))
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :3] = R
                    pose_4x4[:3, 3] = pt_3d
                    poses_3d.append(pose_4x4)

        return poses_3d, shapely_boxes

    def _create_2d_gripper_boxes(self, pt, seg_dir, normal, grid_edge_distance):
        c = self.gripper.config

        half_w1 = (c.e_pg + 2 * (c.i_pg + c.rj)) / 2
        p11 = pt + seg_dir * half_w1
        p12 = p11 + normal * (c.x_pg + c.rj)
        p13 = pt - seg_dir * half_w1 + normal * (c.x_pg + c.rj)
        p14 = pt - seg_dir * half_w1
        rect1 = [p11, p12, p13, p14]

        p21 = pt + seg_dir * half_w1
        p22 = p21 + normal * (c.b_pg + c.c_pg + c.rj)
        p23 = pt - seg_dir * half_w1 + normal * (c.b_pg + c.c_pg + c.rj)
        p24 = pt - seg_dir * half_w1
        rect2 = [p21, p22, p23, p24]

        center_rect3 = pt + normal * (c.b_pg + c.c_pg + c.rj)
        half_base = (c.k_pg + 2 * c.rj) / 2
        height_base = c.d_pg + c.t_pg + c.u_pg + c.rj
        p31 = center_rect3 + seg_dir * half_base
        p32 = p31 + normal * height_base
        p33 = center_rect3 - seg_dir * half_base + normal * height_base
        p34 = center_rect3 - seg_dir * half_base
        rect3 = [p31, p32, p33, p34]

        center_rect4 = center_rect3 + normal * height_base
        half_arm = (max(c.ra, c.rb) + c.re + 2 * c.rj) / 2
        height_arm = c.rc + c.rf + 2 * c.rj
        p41 = center_rect4 + seg_dir * half_arm
        p42 = center_rect4 - seg_dir * half_arm
        p43 = p42 + normal * height_arm
        p44 = p41 + normal * height_arm
        rect4 = [p41, p42, p43, p44]

        half_area = (c.z_pg - 2 * c.rj) / 2
        height_area = c.b_pg - 2 * c.rj
        p51 = pt + seg_dir * half_area
        p52 = p51 + normal * height_area
        p53 = pt - seg_dir * half_area + normal * height_area
        p54 = pt - seg_dir * half_area
        rect5 = [p51, p52, p53, p54]

        center_rect6 = center_rect4 + normal * height_arm
        height_back = grid_edge_distance + c.x_pg + c.rj
        p61 = center_rect6 + seg_dir * half_arm
        p62 = p61 + normal * height_back
        p63 = center_rect6 - seg_dir * half_arm + normal * height_back
        p64 = center_rect6 - seg_dir * half_arm
        rect6 = [p61, p62, p63, p64]

        return [rect1, rect2, rect3, rect4, rect5, rect6]

    def _evaluate_collisions_2d(self, poses_3d, shapely_boxes, poly_lists):
        c = self.gripper.config
        min_area = 0.15 * (c.z_pg - 2 * c.rj) * (c.b_pg - 2 * c.rj)

        polys_p1 = [p for p in poly_lists[0] if not p.is_empty]
        prep_p2 = [prep(p) for p in poly_lists[1] if not p.is_empty]
        prep_p3 = [prep(p) for p in poly_lists[2] if not p.is_empty]
        prep_p4 = [prep(p) for p in poly_lists[3] if not p.is_empty]
        prep_p5 = [prep(p) for p in poly_lists[4] if not p.is_empty]

        valid_poses = []
        valid_areas = []

        stats = {"area": 0, "p2_col": 0, "p3_col": 0, "p4_col": 0, "p5_col": 0}

        for pose, boxes in zip(poses_3d, shapely_boxes):
            rect1, rect2, rect3, rect4, rect5 = boxes[:5]

            area = sum(p.intersection(rect5).area for p in polys_p1)
            if area <= min_area:
                stats["area"] += 1
                continue

            if any(p.intersects(rect3) or p.intersects(rect4) for p in prep_p2):
                stats["p2_col"] += 1
                continue
            if any(p.intersects(rect1) or p.intersects(rect2) for p in prep_p3):
                stats["p3_col"] += 1
                continue
            if any(p.intersects(rect3) or p.intersects(rect4) for p in prep_p4):
                stats["p4_col"] += 1
                continue
            if any(p.intersects(rect4) for p in prep_p5):
                stats["p5_col"] += 1
                continue

            valid_poses.append(pose)
            valid_areas.append(area)

        if len(poses_3d) > 0:
            print(
                f"       -> [Diagnosis] Failed by: Area={stats['area']}, Col(P2)={stats['p2_col']}, Col(P3)={stats['p3_col']}, Col(P4)={stats['p4_col']}, Col(P5)={stats['p5_col']}"
            )

        return valid_poses, valid_areas

    def _calculate_scores(self, valid_poses, valid_areas, pcd):
        c = self.gripper.config
        com = pcd.get_center()

        max_area = max((c.z_pg - 2 * c.rj) * (c.b_pg - 2 * c.rj), 1e-9)
        pts_3d = np.array([pose[:3, 3] for pose in valid_poses])
        dists = np.linalg.norm(pts_3d - com, axis=1)
        max_dist = np.max(dists) if len(dists) > 0 and np.max(dists) > 0 else 1.0

        scores = []
        for pose, area, dist in zip(valid_poses, valid_areas, dists):
            s_area = np.clip((area - 0.15 * max_area) / (0.85 * max_area), 0.0, 1.0)
            s_center = np.clip(1.0 - (dist / max_dist), 0.0, 1.0)
            final_score = (0.9 * s_area) + (0.1 * s_center)
            scores.append((final_score, s_area, s_center, area))

        return scores

    # =========================================================================
    # Visualization
    # =========================================================================
    def visualize_grasp(self, pcd: o3d.geometry.PointCloud, candidate: GraspCandidate):
        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.paint_uniform_color([0.7, 0.7, 0.7])

        gripper_mesh_wrapper = self.gripper.generate_collision_mesh()
        gripper_mesh_o3d = gripper_mesh_wrapper.geometry
        gripper_mesh_o3d.transform(candidate.transform)

        tcp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        tcp_frame.transform(candidate.transform)

        window_title = f"Parallel Grasp | Score: {candidate.score:.3f} | Area: {candidate.score_details['area_score']:.2f}"
        o3d.visualization.draw_geometries(
            [vis_pcd, gripper_mesh_o3d, tcp_frame], window_name=window_title
        )
