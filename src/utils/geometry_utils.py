import copy
from multipledispatch import dispatch
from typing import Any, Dict, Literal, TypedDict, Union
from typing import List, Optional, Tuple

import cv2
from multipledispatch.dispatcher import Dispatcher
import numpy as np
import open3d as o3d
import trimesh
from shapely.geometry import Polygon

from src.utils.types import (
    PointCloud,
    DimensionsReport,
)


# ==============================================================================
# 1. Basic Transformations & Math
# ==============================================================================
def create_pose_matrix(
    rotation_matrix: np.ndarray, translation_vector: np.ndarray
) -> np.ndarray:
    """
    Combines a 3x3 Rotation and 3x1 Translation into a 4x4 Homogeneous Matrix.
    """
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector
    return T


def generate_z_rotation_fan(
    base_rotation: np.ndarray, step_deg: int, range_deg: int
) -> List[np.ndarray]:
    """
    Generates a list of rotation matrices by rotating the 'base_rotation'
    around its own Z-axis in steps.

    Args:
        base_rotation: Initial 3x3 rotation matrix (usually aligned to normal).
        step_deg: Step size in degrees.
        range_deg: Total range (e.g., 180 or 360).

    Returns:
        List of 3x3 rotation matrices.
    """
    rotations = []

    # Avoid infinite loop or zero division
    if step_deg <= 0:
        step_deg = 360

    for angle_deg in range(0, range_deg + 1, step_deg):
        # Create local rotation around Z
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # Standard 3x3 Z-rotation matrix
        R_local_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Apply local rotation on top of base rotation
        # R_final = R_base @ R_local
        R_final = base_rotation @ R_local_z
        rotations.append(R_final)

    return rotations


def create_transformation_matrix(
    rotation: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """
    Creates a 4x4 homogeneous transformation matrix from rotation (axis-angle degrees) and position.

    Args:
        rotation (np.ndarray): [angle_deg, axis_x, axis_y, axis_z]
        position (np.ndarray): [x, y, z] translation vector (usually in meters).

    Returns:
        np.ndarray: 4x4 Transformation Matrix.
    """
    # Normalize position (assuming input might be in mm, logic preserved from original,
    # but explicit division should be handled outside if possible. Keeping as is for compatibility)
    # NOTE: In the original script, position was divided by 1000.0 here.
    # TODO
    # Ideally, units should be handled before calling this.
    # I removed the division to make this a generic geometric utility.
    # ENSURE YOUR INPUT 'position' IS ALREADY IN THE DESIRED UNIT.
    t = np.array(position, dtype=float)

    angle_deg, ax, ay, az = rotation
    angle_rad = np.radians(angle_deg)
    axis = np.array([ax, ay, az], dtype=float)

    # Handle zero vector axis
    if np.linalg.norm(axis) < 1e-9:
        R = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        C = 1 - c

        R = np.array(
            [
                [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
            ]
        )

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def is_parallel(
    normal1: np.ndarray, normal2: np.ndarray, threshold_angle_deg: float = 5.0
) -> bool:
    """
    Checks if two normal vectors are parallel (or anti-parallel) within a threshold.

    Args:
        normal1 (np.ndarray): First normal vector.
        normal2 (np.ndarray): Second normal vector.
        threshold_angle_deg (float): Tolerance in degrees.

    Returns:
        bool: True if parallel.
    """
    # Normalize
    n1 = normal1 / np.linalg.norm(normal1)
    n2 = normal2 / np.linalg.norm(normal2)

    dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle_rad = np.arccos(np.abs(dot_product))  # abs allows anti-parallel check
    return np.degrees(angle_rad) < threshold_angle_deg


def check_normal_alignment(
    plane_normal: np.ndarray, reference_normals: np.ndarray, angle_threshold_deg: float
) -> bool:
    """
    Checks if a single normal vector aligns with ANY vector in a list of references.
    (Renamed from original 'filter_by_normal_orientation' version 1)
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # Normalize references
    refs_norm = (
        reference_normals / np.linalg.norm(reference_normals, axis=1)[:, np.newaxis]
    )

    dot_products = np.dot(plane_normal, refs_norm.T)
    angles_deg = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))

    return np.any(angles_deg <= angle_threshold_deg)


def get_rotation_matrix_between_vectors(
    v_from: np.ndarray, v_to: np.ndarray
) -> np.ndarray:
    """
    Calculates the 3x3 rotation matrix that aligns unit vector v_from to v_to.
    Essential for aligning the Gripper Z-axis to the Surface Normal.

    Args:
        v_from: Source vector (e.g., [0, 0, 1]).
        v_to: Target vector (e.g., surface_normal).
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    # Cross product gives the axis of rotation
    v_cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)

    # Handle edge cases (Already parallel or anti-parallel)
    if np.linalg.norm(v_cross) < 1e-6:
        # If aligned (dot > 0), return Identity.
        # If opposite (dot < 0), return 180 deg rotation (flip).
        return np.eye(3) if dot > 0 else -np.eye(3)

    # Rodrigues Rotation Formula (Skew-symmetric matrix approach)
    s = np.linalg.norm(v_cross)
    c = dot

    vx = np.array(
        [
            [0, -v_cross[2], v_cross[1]],
            [v_cross[2], 0, -v_cross[0]],
            [-v_cross[1], v_cross[0], 0],
        ]
    )

    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
    return R


def apply_transform_to_points(
    points: np.ndarray, transform_matrix: np.ndarray
) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix to an (N, 3) array of points.
    Useful for transforming World Points into the Gripper/Pad Frame.
    """
    if points.shape[1] != 3:
        raise ValueError("Points must be (N, 3)")

    # 1. Convert to Homogeneous coordinates (N, 4) -> Add column of 1s
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])

    # 2. Apply Transform: (T @ P.T).T
    transformed_h = (transform_matrix @ points_h.T).T

    # 3. Drop homogeneous column
    return transformed_h[:, :3]


# ==============================================================================
# 2. Point Cloud Filtering & Pre-processing
# ==============================================================================


def preprocess_point_cloud(
    pcd: PointCloud, voxel_size: float
) -> Tuple[PointCloud, o3d.pipelines.registration.Feature]:
    """
    Downsamples a point cloud, estimates normals, and computes FPFH features.

    Args:
        pcd (PointCloud): Input cloud.
        voxel_size (float): Size of the voxel for downsampling.

    Returns:
        Tuple[PointCloud, Feature]: Downsampled cloud and its FPFH features.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate Normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Compute FPFH Features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return pcd_down, fpfh


def filter_pcd_by_normal_direction(
    pcd: PointCloud,
    n_ref: np.ndarray,
    cos_threshold: float = 0.965,
    radius: Optional[float] = None,
    max_nn: int = 50,
) -> Tuple[PointCloud, np.ndarray]:
    """
    Filters points in a cloud whose normals align with a reference direction.
    (Renamed from original 'filter_by_normal_orientation' version 2)

    Args:
        pcd (PointCloud): Input cloud.
        n_ref (np.ndarray): Reference direction vector.
        cos_threshold (float): Minimum cosine similarity (default ~15 degrees).
        radius (float, optional): Search radius for normal estimation if needed.
        max_nn (int): Max neighbors for normal estimation.

    Returns:
        Tuple[PointCloud, np.ndarray]: Filtered point cloud and boolean mask.
    """
    N = len(pcd.points)
    if N == 0:
        return pcd, np.zeros(0, dtype=bool)

    n_ref = np.asarray(n_ref, dtype=float).reshape(3)
    n_ref = n_ref / np.linalg.norm(n_ref)

    # Estimate normals if missing
    if not pcd.has_normals() or len(pcd.normals) != N:
        if N < 3:
            pcd.normals = o3d.utility.Vector3dVector(
                np.repeat(n_ref[None, :], N, axis=0)
            )
        else:
            if radius is None:
                pts = np.asarray(pcd.points)
                diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
                radius = max(1e-9, 0.02 * diag)

            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=min(max_nn, max(3, N - 1))
                )
            )

    # Orient normals
    try:
        pcd.orient_normals_to_align_with_direction(n_ref)
    except RuntimeError:
        pass  # Fallback to existing orientation

    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)

    # Check alignment (allows parallel and anti-parallel if abs is used,
    # but typically we want directional alignment here. Original code used dot product)
    # NOTE: Original code used `mask = np.abs(cosang) >= float(cos_th)` which allows both directions.
    cosang = np.clip(normals @ n_ref, -1.0, 1.0)
    mask = np.abs(cosang) >= float(cos_threshold)

    out_pcd = pcd.select_by_index(np.where(mask)[0])
    return out_pcd, mask


def remove_outliers_statistical(
    pcd: PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> Tuple[PointCloud, List[int]]:
    """Wraps Open3D statistical outlier removal."""
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return cl, ind


def remove_outliers_dbscan(
    pcd: PointCloud, eps: float = 0.02, min_points: int = 10
) -> PointCloud:
    """
    Clusters the cloud using DBSCAN and returns the largest cluster (main object).
    """
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    if len(labels) == 0:
        return pcd  # Return original if clustering fails

    max_label = labels.max()
    if max_label < 0:
        return pcd  # Only noise found

    # Find largest cluster
    # (Note: label -1 is noise)
    counts = np.bincount(labels[labels >= 0])
    largest_cluster_idx = np.argmax(counts)

    main_cluster_indices = np.where(labels == largest_cluster_idx)[0]
    return pcd.select_by_index(main_cluster_indices)


# ==============================================================================
# 3. Registration (Alignment)
# ==============================================================================


def execute_global_registration(
    source_down: PointCloud,
    target_down: PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Performs RANSAC-based global registration.
    """
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,  # Mutual filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # RANSAC n points
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source: PointCloud,
    target: PointCloud,
    initial_transform: np.ndarray,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Performs ICP (Iterative Closest Point) refinement.
    """
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


# ==============================================================================
# 4. Advanced Geometry (Slicing, Projection, Contours)
# ==============================================================================


def select_points_between_planes(
    pcd: PointCloud,
    center_point: np.ndarray,
    plane_normal: np.ndarray,
    min_dist: float,
    max_dist: float,
) -> Tuple[PointCloud, PointCloud]:
    """
    Slices a point cloud between two parallel planes defined by distance from a center.

    Args:
        pcd: Input cloud.
        center_point: Origin for distance measurement.
        plane_normal: Normal vector defining the slicing direction.
        min_dist: Start distance along the normal.
        max_dist: End distance along the normal.

    Returns:
        Tuple[PointCloud, PointCloud]: (Points Inside Slice, Points Outside)
    """
    points = np.asarray(pcd.points)
    # Project vector from center to points onto the normal to get distance
    distances = (points - center_point) @ plane_normal

    mask = (distances >= min_dist) & (distances <= max_dist)

    pcd_between = pcd.select_by_index(np.where(mask)[0])
    pcd_outside = pcd.select_by_index(np.where(~mask)[0])

    return pcd_between, pcd_outside


def project_points_to_plane(
    pcd: PointCloud, transform_matrix: np.ndarray, plane_normal: np.ndarray
) -> PointCloud:
    """
    Projects points onto a 2D plane defined by the transformation matrix.
    Effectively sets the Z-component (relative to the plane) to zero.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return o3d.geometry.PointCloud()

    # Transform points to the plane's local coordinate system
    # R.T * (P - t) logic or direct multiplication if T is global->local
    # Assuming 'transform_matrix' transforms World -> Local Gripper/Plane Frame

    # Rotate and translate
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3]

    # Note: The original code did: points @ R.T + t
    # This assumes R is the rotation of the gripper in world, and we want to project relative to it?
    # Actually, if we want to flatten, we usually transform World -> Plane Frame.
    # Let's stick to the logic found in the reference script which was:
    # transformed = points @ gripper_transform[:3,:3].T + gripper_transform[:3,3]

    transformed_points = points @ R.T + t

    # Flatten (Project) - Set Z to 0
    # Note: The axis to flatten depends on how the frame is defined.
    # Assuming Z is the depth/approach axis.
    transformed_points[:, 2] = 0

    pcd_proj = o3d.geometry.PointCloud()
    pcd_proj.points = o3d.utility.Vector3dVector(transformed_points)

    # Copy colors if available
    if pcd.has_colors():
        pcd_proj.colors = pcd.colors

    return pcd_proj


def get_plane_contour_polygon(
    pcd_in_plane_coord: PointCloud,
    expansion_value: float = 0.001,
    image_resolution: float = 2000,
) -> Polygon:
    """
    Computes the 2D Shapely Polygon representing the outer contour of a set of points.
    Uses OpenCV for rasterization and contour detection.

    Args:
        pcd_in_plane_coord: Point cloud already projected to 2D (Z=0).
        expansion_value: Amount to buffer/expand the resulting polygon (meters).
        image_resolution: Pixels per meter for rasterization.

    Returns:
        shapely.geometry.Polygon: The resulting 2D polygon.
    """
    if len(pcd_in_plane_coord.points) == 0:
        return Polygon()

    # 1. Extract 2D coordinates
    points_2d = np.asarray(pcd_in_plane_coord.points)[:, :2]

    if len(points_2d) < 3:
        return Polygon()

    # 2. Calculate Bounding Box for Image Creation
    min_x, min_y = points_2d.min(axis=0)
    max_x, max_y = points_2d.max(axis=0)

    width = int((max_x - min_x) * image_resolution) + 20  # +padding
    height = int((max_y - min_y) * image_resolution) + 20

    if width <= 0 or height <= 0:
        return Polygon()

    # 3. Create Image and Draw Points
    img = np.zeros((height, width), dtype=np.uint8)

    # Shift points to image coords
    # (x - min_x) * scale + padding
    pts_img = ((points_2d - [min_x, min_y]) * image_resolution).astype(np.int32) + 10

    # Draw filled polygon based on points (convex hull of local points or just dots?)
    # Original script used cv2.fillPoly on the raw points?
    # If points are scattered, fillPoly might behave oddly if not ordered.
    # Usually one draws points then morphology.
    # Replicating original logic:
    cv2.fillPoly(img, [pts_img], color=255)

    # 4. Find Contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return Polygon()

    # Take the largest contour
    cnt = max(contours, key=cv2.contourArea)

    # 5. Convert back to World Coordinates
    # (pixel - padding) / scale + min
    contour_world = (cnt.reshape(-1, 2) - 10) / image_resolution + [min_x, min_y]

    if len(contour_world) < 3:
        return Polygon()

    poly = Polygon(contour_world)

    # 6. Simplify and Buffer
    poly = poly.simplify(0.001, preserve_topology=True)
    if expansion_value > 0:
        poly = poly.buffer(expansion_value)

    return poly


def merge_coplanar_planes(
    pcd: PointCloud, distance_threshold: float = 0.002, plane_angle_thresh: float = 5.0
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Iteratively segments planes from a point cloud using RANSAC and merges similar ones.

    Args:
        pcd: Input cloud.
        distance_threshold: RANSAC distance threshold.
        plane_angle_thresh: Degrees to consider planes parallel.

    Returns:
        List of tuples: [(plane_model_eq, plane_center, plane_normal), ...]
    """
    # NOTE: This logic was embedded in the main loop of the original script.
    # This is a simplified utility version. Full merging logic is complex.
    # For now, this wrapper suggests using Open3D's segment_plane iteratively.

    planes_found = []
    working_pcd = copy.deepcopy(pcd)

    # Safety break to prevent infinite loops
    max_planes = 100

    for _ in range(max_planes):
        if len(working_pcd.points) < 50:
            break

        plane_model, inliers = working_pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
        )

        # Extract plane normal
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # Extract points
        plane_pcd = working_pcd.select_by_index(inliers)
        center = plane_pcd.get_center()

        # Save result
        planes_found.append((plane_model, center, normal))

        # Remove inliers to find next plane
        working_pcd = working_pcd.select_by_index(inliers, invert=True)

    return planes_found


def compute_pcd_normals(
    pcd: o3d.geometry.PointCloud,
    radius: Optional[float] = None,
    max_nn: int = 30,
    align_vector: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """
    Estimates normals for a point cloud.

    Args:
        pcd: Input point cloud.
        radius: Search radius. If None, calculated automatically from bounding box.
        max_nn: Max neighbors to use.
        align_vector: Optional [x,y,z] vector to orient normals towards (e.g., [0,0,1]).
    """
    # 1. Automatic Radius Calculation if not provided
    if radius is None:
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return pcd
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        radius = max(1e-9, 0.02 * diag)  # Heuristic: 2% of diagonal

    # 2. Estimate Normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    # 3. Orient Normals (Critical for Poisson and Vacuum Gripper)
    if align_vector is not None:
        # Orient consistent with a tangent plane first to fix local flips
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Then align globally to the specific direction
        vec = np.asarray(align_vector, dtype=float)
        try:
            pcd.orient_normals_to_align_with_direction(vec)
        except RuntimeError:
            pass

    return pcd


####---
def create_cylinder(
    length: float,
    radius: float,
    start_point: np.ndarray = np.array([0.0, 0.0, 0.0]),
    direction_vector: np.ndarray = np.array([0.0, 0.0, 1.0]),
    color: Optional[List[float]] = None,
    backend: Literal["trimesh", "open3d"] = "trimesh",
) -> Union[trimesh.Trimesh, o3d.geometry.TriangleMesh]:
    """
    Generate a cylinder mesh using the specified backend.

    Args:
        backend: "trimesh" or "open3d".
        color: List RGB [0.0-1.0].
    """
    if backend == "trimesh":
        return create_cylinder_trimesh(
            length=length,
            radius=radius,
            start_point=start_point,
            direction_vector=direction_vector,
            color=color,
        )
    else:
        return create_cylinder_o3d(
            length=length,
            radius=radius,
            start_point=start_point,
            direction_vector=direction_vector,
            color=color,
        )


def create_cylinder_trimesh(
    length: float,
    radius: float,
    start_point: np.ndarray = np.array([0.0, 0.0, 0.0]),
    direction_vector: np.ndarray = np.array([0.0, 0.0, 1.0]),
    color: Optional[List[float]] = None,
) -> trimesh.Trimesh:
    """
    Generate a cylinder Trimesh oriented along a direction vector.

    Args:
        color: Lista RGBA [0-255].
    """
    # 1. Criar cilindro básico (alinhado em Z, centro na origem 0,0,0)
    mesh = trimesh.creation.cylinder(radius=radius, height=length)

    # 2. Calcular matriz de rotação para alinhar Z com o vetor de direção
    # O trimesh devolve uma matriz 4x4 completa
    z_axis = np.array([0, 0, 1])
    T_rot = trimesh.geometry.align_vectors(z_axis, direction_vector)

    # 3. Aplicar a rotação
    mesh.apply_transform(T_rot)

    # 4. Translação
    # O cilindro agora está rodado, mas ainda centrado na origem (0,0,0).
    # Precisamos movê-lo para que a BASE fique no start_point.
    # O centro final deve ser: Start + (Direção Normalizada * Metade do Comprimento)
    vec_norm = direction_vector / np.linalg.norm(direction_vector)
    final_center = start_point + (vec_norm * (length / 2.0))

    # Como o mesh está em 0,0,0, basta aplicar uma translação direta
    T_trans = trimesh.transformations.translation_matrix(final_center)
    mesh.apply_transform(T_trans)

    # 5. Cor
    if color is not None:
        # CONVERSION LOGIC:
        # Input is 0.0-1.0. Trimesh wants 0-255.
        # We simply multiply by 255 and cast to integer.
        color_u8 = [int(c * 255) for c in color]
        mesh.visual.face_colors = color_u8

    return mesh


def create_cylinder_o3d(
    length: float,
    radius: float,
    start_point: np.ndarray = np.array([0.0, 0.0, 0.0]),
    direction_vector: np.ndarray = np.array([0.0, 0.0, 1.0]),
    color: Optional[List[float]] = None,
) -> o3d.geometry.TriangleMesh:
    """
    Gera um cilindro Open3D orientado.

    Args:
        color: Lista RGB [0.0-1.0].
    """
    # 1. Criar cilindro (alinhado em Z, centro na origem)
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

    # 2. Obter matriz de rotação via Trimesh
    z_axis = np.array([0, 0, 1])
    T_rot = trimesh.geometry.align_vectors(z_axis, direction_vector, return_angle=False)

    # O Open3D usa apenas a parte 3x3 para rotação (R)
    R = T_rot[:3, :3]

    # 3. Aplicar rotação (em torno do centro 0,0,0)
    mesh.rotate(R, center=np.array([0, 0, 0]))

    # 4. Translação para o centro final
    vec_norm = direction_vector / np.linalg.norm(direction_vector)
    final_center = start_point + (vec_norm * (length / 2.0))

    mesh.translate(final_center)

    # 5. Cor
    if color is not None:
        # SANITIZATION LOGIC:
        # Input is [R, G, B, A]. Open3D only supports [R, G, B].
        # We take the first 3 elements.
        color_rgb = color[:3]
        mesh.paint_uniform_color(color_rgb)

    return mesh


###---
def _calculate_report_from_extents(
    extents: np.ndarray, type_name: str
) -> DimensionsReport:
    """
    Internal helper to calculate statistics and generate the report
    once we have the bounding box extents.
    """
    # 1. Calculate Diagonal
    diagonal = float(np.linalg.norm(extents))

    # 2. Heuristic Unit Estimation
    # Logic: Diagonal > 50.0 implies millimeters (e.g., 250mm can).
    # Otherwise, assumes meters.
    likely_unit: Literal["meters", "millimeters"] = "meters"
    if diagonal > 50.0:
        likely_unit = "millimeters"

    # 3. Suggest Voxel Size (~1/100 of size)
    suggested_voxel_size = diagonal / 100.0

    # 4. Build Report
    return DimensionsReport(
        **{
            "extents_xyz": np.round(extents, 4),
            "diagonal": round(diagonal, 4),
            "likely_unit": likely_unit,
            "suggested_voxel_size": round(suggested_voxel_size, 5),
            "details": (
                f"Type: {type_name}. "
                f"Size: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}. "
                f"Unit: {likely_unit.upper()}."
            ),
        }
    )


def load_mesh_file(path: str) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Safely loads an external 3D asset (.stl, .obj) and ensures it has normals.
    """
    try:
        # Open3D handles format detection automatically
        mesh = o3d.io.read_triangle_mesh(path)

        if not mesh.has_triangles():
            print(f"[GeometryUtils] Warning: Loaded mesh '{path}' has no triangles.")
            return None

        # Critical for good visualization
        mesh.compute_vertex_normals()
        return mesh

    except Exception as e:
        print(f"[GeometryUtils] Error loading mesh '{path}': {e}")
        return None


def merge_meshes(
    meshes: Union[List[trimesh.Trimesh], List[o3d.geometry.TriangleMesh]],
) -> Union[trimesh.Trimesh, o3d.geometry.TriangleMesh]:
    """
    Combina uma lista de malhas em uma única malha,
    detectando automaticamente se é Trimesh ou Open3D.
    """
    if not meshes:
        raise ValueError("A lista de malhas está vazia.")

    # Verifica o tipo do primeiro elemento para decidir a estratégia
    first_mesh = meshes[0]

    # --- Estratégia Trimesh ---
    if isinstance(first_mesh, trimesh.Trimesh):
        # trimesh.util.concatenate é super eficiente
        return trimesh.util.concatenate(meshes)

    # --- Estratégia Open3D ---
    elif isinstance(first_mesh, o3d.geometry.TriangleMesh):
        # Open3D usa soma de operadores para merge simples
        merged = meshes[0]
        for mesh in meshes[1:]:
            merged += mesh
        merged.compute_vertex_normals()
        return merged

    else:
        raise TypeError(f"Tipo de malha não suportado para merge: {type(first_mesh)}")


# --- Main Dispatch Functions ---
analyze_object_dimensions = Dispatcher("analyze_object_dimensions")


@analyze_object_dimensions.register(trimesh.Trimesh)
def _analyze_trimesh(geometry: trimesh.Trimesh) -> DimensionsReport:
    """Trimesh Implementation"""
    return _calculate_report_from_extents(geometry.extents, "Trimesh")


@analyze_object_dimensions.register(o3d.geometry.PointCloud)
def _analyze_pcd(geometry: o3d.geometry.PointCloud) -> DimensionsReport:
    """Open3D PointCloud Implementation"""
    aabb = geometry.get_axis_aligned_bounding_box()
    return _calculate_report_from_extents(aabb.get_extent(), "Open3D PointCloud")


@analyze_object_dimensions.register(o3d.geometry.TriangleMesh)
def _analyze_o3d_mesh(geometry: o3d.geometry.TriangleMesh) -> DimensionsReport:
    """Open3D Mesh Implementation"""
    aabb = geometry.get_axis_aligned_bounding_box()
    return _calculate_report_from_extents(aabb.get_extent(), "Open3D Mesh")


# Default
@analyze_object_dimensions.register(object)
def _analyze_default(geometry: object) -> DimensionsReport:
    raise TypeError(f"Non-supported type: {type(geometry)}")


class PointCloudDensityReport(TypedDict):
    average_distance: float
    median_distance: float
    mode_distance: float


def get_point_cloud_density(pcd: o3d.geometry.PointCloud) -> PointCloudDensityReport:
    """
    Calculates the average distance between nearest neighbors in the point cloud.

    This metric is essential for automatically determining robust parameters
    for algorithms like DBSCAN (eps) and Normal Estimation (search radius),
    making the pipeline independent of the camera distance/resolution.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.

    Returns:
        float: The mean distance between points (in the same unit as the cloud, e.g., meters).
    """
    # Safety check for empty clouds
    if not pcd.has_points():
        return {
            "average_distance": 0.0,
            "median_distance": 0.0,
            "mode_distance": 0.0,
        }

    # Open3D built-in method to find the distance to the closest neighbor for every point
    distances = pcd.compute_nearest_neighbor_distance()

    # We use the arithmetic mean to estimate density.
    # Note: Median can be used for outlier robustness, but mean is standard for this purpose.
    avg_dist = np.mean(distances)
    median_dist = np.median(distances)
    mode_dist = (
        float(np.bincount(np.round(np.array(distances) * 1e6).astype(int)).argmax())
        / 1e6
    )
    dict_dists: PointCloudDensityReport = {
        "average_distance": float(avg_dist),
        "median_distance": float(median_dist),
        "mode_distance": float(mode_dist),
    }

    return dict_dists


def create_raycasting_scene_from_hull(
    pcd: o3d.geometry.PointCloud,
) -> o3d.t.geometry.RaycastingScene:
    """
    Computes the Convex Hull of a Point Cloud and sets up an Open3D RaycastingScene.
    """
    hull_mesh, _ = pcd.compute_convex_hull()

    try:
        # Open3D Tensor Geometry is required for Raycasting
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(hull_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        return scene
    except Exception as e:
        raise Exception(f"[GeometryUtils] Error creating RaycastingScene: {e}")


def generate_fibonacci_sphere_points(n_samples: int) -> np.ndarray:
    """
    Generates N evenly distributed unit vectors on a sphere (Fibonacci Lattice).
    Returns: np.ndarray shape (N, 3)
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(n_samples):
        y = 1 - (i / float(n_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])

    return np.array(points)


def generate_inward_rays_from_sphere(
    center: np.ndarray, radius: float, n_samples: int
) -> List[np.ndarray]:
    """
    Generates rays starting from a bounding sphere pointing towards the center.
    Returns: List of arrays [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
    """
    # Get distributed directions (Unit vectors pointing OUT)
    sphere_dirs = generate_fibonacci_sphere_points(n_samples)
    rays_list = []

    for dir_vec in sphere_dirs:
        # Origin on sphere surface
        origin = center + (dir_vec * radius)

        # Direction towards center (normalized)
        # cast_dir = center - origin
        # cast_dir = cast_dir / np.linalg.norm(cast_dir)
        cast_dir = -dir_vec  # Since dir_vec is already unit length

        # Pack into 6D vector for Open3D
        ray = np.concatenate([origin, cast_dir]).astype(np.float32)
        rays_list.append(ray)

    return rays_list


def get_nearest_point_in_cloud(
    query_point: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    pcd_tree: o3d.geometry.KDTreeFlann,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the closest point and normal in the point cloud to the query coordinate.
    Returns: (point, normal)
    """
    # search_knn_vector_3d returns [k, indices, distances]
    [_, idx_list, _] = pcd_tree.search_knn_vector_3d(query_point, 1)
    real_idx = idx_list[0]

    real_point = np.asarray(pcd.points[real_idx])
    real_normal = np.asarray(pcd.normals[real_idx])

    return real_point, real_normal


def trimesh_to_open3d(mesh_trimesh) -> o3d.geometry.TriangleMesh:
    """
    Converts a Trimesh object to an Open3D TriangleMesh.
    """
    # 1. Create empty Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()

    # 2. Copy Vertices
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)

    # 3. Copy Faces (Triangles)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)

    # 4. Compute Normals (Essential for lighting/rendering)
    mesh_o3d.compute_vertex_normals()

    return mesh_o3d


def create_score_heatmap_pcd(
    points: List[np.ndarray], scores: List[float]
) -> o3d.geometry.PointCloud:
    """
    Creates a Point Cloud where points are colored based on their score.

    Gradient Strategy (Traffic Light):
    - Score 0.0 (Bad)  -> Red   [1.0, 0.0, 0.0]
    - Score 0.5 (Mid)  -> Yellow[1.0, 1.0, 0.0]
    - Score 1.0 (Good) -> Green [0.0, 1.0, 0.0]
    """
    heatmap_pcd = o3d.geometry.PointCloud()

    if not points:
        return heatmap_pcd

    # 1. Set Geometry
    heatmap_pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # 2. Calculate Colors
    colors = []
    for s in scores:
        # Clamp score 0.0-1.0
        val = max(0.0, min(1.0, s))

        # Traffic Light Math:
        # - Low Score: High Red, Low Green
        # - High Score: Low Red, High Green
        # r = 1.0 when val < 0.5, then drops to 0.0
        # g = 0.0 when val < 0.0, then rises to 1.0

        r = min(1.0, 2.0 * (1.0 - val))
        g = min(1.0, 2.0 * val)
        b = 0.0

        colors.append([r, g, b])

    heatmap_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return heatmap_pcd


def create_ring_mesh(
    radius: float, thickness: float, height: float, relative: bool
) -> trimesh.Trimesh:
    """
    Create a 3D mesh representing a ring (hollow cylinder).

    Args:
        radius (float): Outer radius of the ring.
        thickness (float): Thickness of the ring walls.
        height (float): Height of the ring.
        relative (bool): Whether the thickness is relative to the radius.

    Returns:
        trimesh.Trimesh: The resulting geometry.
    """
    # Calculate inner radius
    if relative:
        inner_radius = radius * (1 - thickness)
    else:
        inner_radius = radius - thickness

    inner_radius = max(inner_radius, 0)  # Ensure non-negative radius

    outer_mesh = trimesh.creation.cylinder(radius=radius, height=height)
    inner_mesh = trimesh.creation.cylinder(radius=inner_radius, height=height)

    return outer_mesh.difference(inner_mesh)


def create_wall_mesh(
    width: float, height: float, thickness: float, depth: float, relative: bool
) -> trimesh.Trimesh:
    """
    Create a 3D mesh representing a rectangular frame with hollowed-out walls.

    Args:
        width (float): Outer width of the box.
        height (float): Outer height of the box.
        thickness (float): Thickness of the walls.
        depth (float): Depth of the box (length in the z-direction).
        relative (bool): Whether the thickness is relative to the dimensions.

    Returns:
        trimesh.Trimesh: The resulting geometry.
    """
    # Calculate inner dimensions
    if relative:
        inner_width = width * (1 - thickness)
        inner_height = height * (1 - thickness)
    else:
        inner_width = width - 2 * thickness
        inner_height = height - 2 * thickness

    inner_width = max(inner_width, 0)  # Ensure non-negative dimensions
    inner_height = max(inner_height, 0)

    outer_mesh = trimesh.creation.box(extents=[width, height, depth])
    inner_mesh = trimesh.creation.box(extents=[inner_width, inner_height, depth])

    return outer_mesh.difference(inner_mesh)
    return heatmap_pcd
