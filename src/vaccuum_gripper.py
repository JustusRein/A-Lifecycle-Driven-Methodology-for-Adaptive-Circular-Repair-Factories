import open3d as o3d
import numpy as np
import random
import copy
from math import acos, degrees
from sklearn.decomposition import PCA
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import cv2
import itertools
from shapely.ops import unary_union
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.validation import explain_validity, make_valid  # Shapely>=2.0
from shapely import set_precision
import os

# -------------------- Define the gripper ----------------

# For unit change from mm to m
# MM_TO_M = 0.001

# def load_yaml_in_m(filepath):
#     """Read YAML file，change unit from mm to m"""
#     with open(filepath, "r", encoding="utf-8") as f:
#         data_mm = yaml.safe_load(f)

#     # change to m
#     data_m = {
#         key: (value * MM_TO_M if isinstance(value, (int, float)) else value)
#         for key, value in data_mm.items()
#     }
#     return data_m

# # mm to m
# params = load_yaml_in_m(r"gripper_parameter\Franka.yaml")


with open(os.path.join("gripper_parameter", "Franka.yaml"), "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


a_pg = params["a_pg"]  # Finger width
w_pg = params["w_pg"]  # Internal Safespace Finger width
v_pg = params["v_pg"]  # External Safespace Finger width
f_pg = params["f_pg"]  # Distance gripper open
g_pg = params["g_pg"]  # Distance gripper close
h_pg = params["h_pg"]  # Gripper base bottom width
k_pg = params["k_pg"]  # Safespace Gripper base bottom width
q_pg = params["q_pg"]  # Gripper base top width
r_pg = params["r_pg"]  # Safespace Gripper base top width

y_pg = max(
    q_pg + 2 * r_pg, h_pg + 2 * k_pg, f_pg + 2 * (a_pg + v_pg)
)  # Gripper Bounding box max width

b_pg = params["b_pg"]  # Gripper area length end
c_pg = params["c_pg"]  # Gripper area to (Safety space of Gripper)length end
d_pg = params["d_pg"]  # Safespace Gripper length
x_pg = params["x_pg"]  # Safespace Gripper end to rubber
n_pg = d_pg + c_pg + b_pg  # Finger length
t_pg = params["t_pg"]  # Gripper base bottom length
u_pg = params["u_pg"]  # Gripper base top length
s_pg = n_pg + t_pg + u_pg + x_pg  # Total gripper length

e_pg = params["e_pg"]  # Finger depth
i_pg = params["i_pg"]  # Safespace finger depth

z_pg = params["z_pg"]  # Gripper area depth

l_pg = params["l_pg"]  # Gripper base bottom depth
m_pg = params["m_pg"]  # Safespace gripper base bottom depth
o_pg = params["o_pg"]  # Gripper base top  depth
p_pg = params["p_pg"]  # Safespace gripper base top depth

j_pg = max(
    l_pg + 2 * m_pg, o_pg + 2 * p_pg, e_pg + 2 * i_pg
)  # Gripper Bounding box max depth


ra = params["ra"]  # width of last robot arm limb
rb = params["rb"]  # depth of last robot arm limb
rc = params["rc"]  # length of last robot arm limb
rd = max(ra, rb)  # maximum diameter of last robot arm limb
re = params["re"]  # robot arm diameter clearance
rf = params["rf"]  # robot arm length clearance
rj = params["rj"]  # repeatability of robot arm

print("Gripper parameters:")
print(f"a_pg: {a_pg:.3f} m")
print(f"w_pg: {w_pg:.3f} m")

# ***************** Code parameters **********

plane_angle_thresh = 5.0

min_remaining_points = 50
min_points_per_plane = 50  # for plane segment clustering

distance_threshold = 1  # plane segment distance threshold
max_planes = 200  # maximum number of planes to detect

margin_points_between_planes = 1


tilt_symbol_start_dist = 18
tilt_symbol_handle_length = 15
tilt_symbol_finger_width_half = 5
tilt_symbol_finger_end_length = 8

plt_graphic_padding = 10
contour_image_padding = 10

# ********* image options ****************
no_image = False
essential_image_only = False
no_skip = True

show_all_planes_and_normals = True
show_planes_parallel_clustering = True
show_plane_pairs = True
show_plane_pair_and_proj_in_pcd = True
show_proj_pts_p1 = True
show_proj_pts_p2 = True
show_proj_pts_p3 = True
show_proj_pts_p4 = True
show_proj_pts_p5 = True
show_P2345_in_pcd = True
show_each_P_in_pcd = True
show_plt_contour_P2_2d = True
show_P2_contour_3d = True
show_plt_all_tcp_grids = False
show_plt_TCP_each_edge = True
show_plt_bounding_boxes = True
show_plt_contours_Px_2d = True
show_P_contour_3d = True
show_feasible_each_edge = True
show_all_feasbile_in_2d = True
show_feasible_with_P_and_pcd = True

if no_image:
    no_skip = True


if essential_image_only:
    show_all_planes_and_normals = True
    show_planes_parallel_clustering = True
    show_plane_pairs = True
    show_plane_pair_and_proj_in_pcd = True
    show_proj_pts_p2 = True
    show_proj_pts_p1 = True
    show_proj_pts_p3 = True
    show_proj_pts_p4 = True
    show_proj_pts_p5 = True
    show_P2345_in_pcd = True
    show_each_P_in_pcd = False
    show_plt_contour_P2_2d = True
    show_P2_contour_3d = True
    show_plt_all_tcp_grids = True
    show_plt_TCP_each_edge = True
    show_plt_bounding_boxes = False
    show_plt_contours_Px_2d = True
    show_P_contour_3d = True
    show_feasible_each_edge = True
    show_all_feasbile_in_2d = True
    show_feasible_with_P_and_pcd = True

# *******************************************


# **************************** Aid Functions **************************************
def filter_by_normal_orientation(
    pcd, n_ref, cos_th=0.965, knn=30, radius=None, max_nn=50
):
    N = len(pcd.points)
    if N == 0:
        return pcd, np.zeros(0, dtype=bool)

    n_ref = np.asarray(n_ref, dtype=float).reshape(3)
    nr = np.linalg.norm(n_ref)
    if nr == 0 or not np.isfinite(nr):
        raise ValueError("n_ref must be a non-zero, finite vector")
    n_ref = n_ref / nr

    need_est = (not pcd.has_normals()) or (len(pcd.normals) != N)

    if need_est:
        if N < 3:
            # Too few points to estimate using neighborhood PCA; directly assign known plane normal vectors.
            pcd.normals = o3d.utility.Vector3dVector(
                np.repeat(n_ref[None, :], N, axis=0)
            )
        else:
            # Radius adaptive stabilization (adaptive to scale changes)
            if radius is None:
                pts = np.asarray(pcd.points)
                diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
                radius = max(
                    1e-9, 0.02 * diag
                )  # Experience points: ~2% of the diagonal
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=min(max_nn, max(3, N - 1))
                )
            )
            # Check again if normals are estimated correctly
            if (not pcd.has_normals()) or (len(pcd.normals) != N):
                pcd.normals = o3d.utility.Vector3dVector(
                    np.repeat(n_ref[None, :], N, axis=0)
                )

    # Unify direction, fail to fall back to direct assignment
    try:
        pcd.orient_normals_to_align_with_direction(n_ref)
    except RuntimeError:
        pcd.normals = o3d.utility.Vector3dVector(np.repeat(n_ref[None, :], N, axis=0))

    pcd.normalize_normals()

    normals = np.asarray(pcd.normals)
    cosang = np.clip(normals @ n_ref, -1.0, 1.0)

    mask = np.abs(cosang) >= float(cos_th)
    out = pcd.select_by_index(np.where(mask)[0])
    return out, mask


def remove_pcd_outlier_statistical(pcd, neighbots=20, std_ratio=1.0):
    if len(pcd.points) == 0:
        print("Remove Outlier Error: Input point cloud is empty.")
        return pcd, None

    # statitical
    if isinstance(pcd, o3d.geometry.PointCloud):
        filtered, ind = pcd.remove_statistical_outlier(
            nb_neighbors=neighbots, std_ratio=std_ratio
        )
        return filtered, ind
    elif isinstance(pcd, np.ndarray):
        pcloud = o3d.geometry.PointCloud()
        pcloud.points = o3d.utility.Vector3dVector(pcd)
        filtered, ind = pcloud.remove_statistical_outlier(
            nb_neighbors=neighbots, std_ratio=std_ratio
        )
        return np.asarray(filtered.points), ind
    else:
        print(
            "Error: Input type is not supported, neither 'PointCloud' nor 'np.ndarray'."
        )
        return None, None


def remove_pcd_outlier_dbscan(
    pcd, eps=0.007, min_samples=20, min_cluster_ratio=0.02, verbose=True
):
    if len(pcd.points) <= 500:
        pcd_null = o3d.geometry.PointCloud()
        pcd_null.points = o3d.utility.Vector3dVector([])
        return pcd_null, None
    else:
        return pcd, None
        # return pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)[0],None


# *****************
# *************** Read and merge part point clouds ***************
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.remove_non_finite_points()
    return pcd


def merge_part_pcd(pcd_paths):
    all_points = []
    all_colors = []
    all_part_ids = []
    pcd_parts_list = []

    for part_id, path in enumerate(pcd_paths):
        pcd = load_point_cloud(path)

        if not pcd.has_colors():
            color = [random.random(), random.random(), random.random()]  # Random color
            pcd.paint_uniform_color(color)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        part_ids = np.full((points.shape[0], 1), part_id)

        all_points.append(points)
        all_colors.append(colors)
        all_part_ids.append(part_ids)
        pcd_parts_list.append(pcd)

    # Merge all parts into one point cloud
    points_assembly = np.vstack(all_points)
    colors_assembly = np.vstack(all_colors)
    part_ids_assembly = np.vstack(all_part_ids)

    pcd_assembly = o3d.geometry.PointCloud()
    pcd_assembly.points = o3d.utility.Vector3dVector(points_assembly)
    pcd_assembly.colors = o3d.utility.Vector3dVector(colors_assembly)

    o3d.visualization.draw_geometries(
        [pcd_assembly], window_name="Merged Point Cloud Assembly"
    )

    return (
        pcd_assembly,
        points_assembly,
        colors_assembly,
        part_ids_assembly,
        pcd_parts_list,
    )


# ******************* Registration and Missing Point Detection **************


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return pcd_down, fpfh


def execute_global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )


def refine_registration(source, target, init_trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )


def find_missing_points(model_pcd, actual_pcd, distance_threshold=0.005):
    actual_tree = o3d.geometry.KDTreeFlann(actual_pcd)
    missing_points = []
    missing_points_indices = []

    for i, pt in enumerate(model_pcd.points):
        [k, idx, _] = actual_tree.search_knn_vector_3d(pt, 1)
        nearest = np.asarray(actual_pcd.points)[idx[0]]
        if np.linalg.norm(pt - nearest) > distance_threshold:
            missing_points.append(pt)
            missing_points_indices.append(i)

    missing_pcd = o3d.geometry.PointCloud()

    if len(missing_points) > 0:
        missing_pcd.points = o3d.utility.Vector3dVector(np.array(missing_points))
        missing_pcd.paint_uniform_color([1, 0, 0])
    else:
        print("[Info] No Missing Points Found")

    return missing_points_indices, missing_pcd


def visualize_regis_and_miss_pts_result(model, actual, missing_pts_indices):
    colors = np.asarray(model.colors)
    model.paint_uniform_color([0.5, 0.5, 0.5])
    actual.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries(
        ([model, actual]),
        window_name="Registration Result of Scaned and Model Point Cloud",
    )
    model.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries(
        [model.select_by_index(missing_pts_indices), actual],
        window_name="Missing Point of Model Point Cloud",
    )
    model.colors = o3d.utility.Vector3dVector(colors)


# ****************** Detect the Source Part of Missing Points **************


def find_source_part_of_missing_points(
    missing_points_indices, part_ids_assembly, pcd_parts_list, miss_perc_thresh=25.0
):
    max_num = int(np.max(part_ids_assembly))
    missing_amount = np.zeros(max_num + 1)
    miss_percentages = []
    part_points_num_list = [len(pcd.points) for pcd in pcd_parts_list]
    detect_target_list = []
    for i in missing_points_indices:
        index = part_ids_assembly[i]
        missing_amount[index] += 1
    for i in range(max_num + 1):
        miss_percentages.append((missing_amount[i] / part_points_num_list[i]) * 100.0)
    for i in range(max_num + 1):
        print(
            f"Part {i + 1} Missing Points Amount: {missing_amount[i]}, Missing Percentage: {miss_percentages[i]:.2f}%"
        )
        if miss_percentages[i] > miss_perc_thresh:
            print(f"Part {i + 1} missing over 30% of points, Do TCP Detection for it.")
            detect_target_list.append(i)
    if len(detect_target_list) == 0:
        print("No Part need to be replaced.")
        exit()
    return detect_target_list


# ****************

# ********paras*****
miss_perc_thresh = 25.0
# ******************

pcd_paths = [
    os.path.join(r"Test_part", "Verification_examples", "01_Bottom_CubeSat_MODEL.pcd"),
    os.path.join(r"Test_part", "Verification_examples", "02_Wall_no_Logo_1.pcd"),
    os.path.join(r"Test_part", "Verification_examples", "02_Wall_with_acor_logo.pcd"),
    os.path.join(
        r"Test_part",
        "Verification_examples",
        "02_wall_with_PLCM_logo_plcm_02_Wall_no_Logo.pcd",
    ),
    os.path.join(
        r"Test_part",
        "Verification_examples",
        "02_wall_with_PLCM_logo_plcm_PLCM_Logo.pcd",
    ),
    os.path.join(r"Test_part", "Verification_examples", "03_Board_Empty.pcd"),
]


pcd_assembly, points_assembly, colors_assembly, part_ids_assembly, pcd_parts_list = (
    merge_part_pcd(pcd_paths)
)
actual_pcd = load_point_cloud(
    os.path.join(r"Test_part", "Verification_examples", "Scanned_pcd.pcd")
)

actual_pcd.scale(0.001, np.array([0, 0, 0]))  # Scale

voxel_size = 0.007  # voxel down radius, adjust regarding the dense of pcd

# Down voxel, compute fpfh
model_down, model_fpfh = preprocess_point_cloud(pcd_assembly, voxel_size)
actual_down, actual_fpfh = preprocess_point_cloud(actual_pcd, voxel_size)

# RANSAC Registration
result_ransac = execute_global_registration(
    model_down, actual_down, model_fpfh, actual_fpfh, voxel_size
)

# ICP Registration
radius_normal = voxel_size * 2
pcd_assembly.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
)
actual_pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
)
result_icp = refine_registration(
    pcd_assembly, actual_pcd, result_ransac.transformation, voxel_size
)
pcd_assembly.transform(result_icp.transformation)
for i in range(len(pcd_parts_list)):
    pcd_parts_list[i].transform(result_icp.transformation)
    # o3d.visualization.draw_geometries([actual_pcd,pcd_parts_list[i]],window_name=f"Part {i+1} after ICP Registration")


missing_point_indices, missing_pcd = find_missing_points(
    pcd_assembly, actual_pcd, 0.0015
)

visualize_regis_and_miss_pts_result(pcd_assembly, actual_pcd, missing_point_indices)


detect_target_list = find_source_part_of_missing_points(
    missing_point_indices, part_ids_assembly, pcd_parts_list, miss_perc_thresh
)

# ***************************************************************************
# for part in detect_target_list:
#     pcd_target = pcd_parts_list[part]

# All the following code is for TCP detection for one damaged part.
# The following code should be performed once for each part listed in detect_target_list.
# Here we set "part = 2", Using "Wall with acor logo" as an example, the code will only execute once for this part to show the result.
# To perform TCP detection for all damaged parts, tthe following code should be added to the for loop above.

part = 2
pcd_target = pcd_parts_list[part]

# ***************** Find the Points on the Scanned Part that are Close to the Target Part ******************


def scanned_points_in_part_radius(
    part_pcd: np.ndarray, scanned_pcd: np.ndarray, r: float
):
    """
    part_pcd: (N_part, 3) numpy array, reference point cloud (the part)
    scanned_pcd: (N_scan, 3) numpy array, test point cloud (the scan)
    r: radius threshold (same unit as the point cloud, e.g. meters or millimeters)

    Returns:
        scan_in:   (M, 3) points from scanned_pcd that fall within radius r of part_pcd
        scan_out:  (N_scan - M, 3) remaining points from scanned_pcd
        mask:      (N_scan,) boolean array, True if the scan point is within r of any part point
        nn_indices: indices of the nearest part points for the inlier scan points
        nn_dists2: squared distances to the nearest part points
    """
    assert part_pcd.ndim == 2 and part_pcd.shape[1] == 3
    assert scanned_pcd.ndim == 2 and scanned_pcd.shape[1] == 3
    assert r > 0

    # Build Open3D point cloud and KD-tree (indexing part_pcd)
    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(part_pcd)
    kdtree = o3d.geometry.KDTreeFlann(pcdA)

    mask = np.zeros(len(scanned_pcd), dtype=bool)
    # Optional: store nearest neighbor index and distance
    nn_indices = -np.ones(len(scanned_pcd), dtype=int)
    nn_dists2 = np.full(len(scanned_pcd), np.inf, dtype=float)

    # For each scanned point, check if it lies within radius r of any part point
    for i, pb in enumerate(scanned_pcd):
        k, idxs, dists2 = kdtree.search_radius_vector_3d(pb, r)
        if k > 0:
            mask[i] = True
            # Record nearest part point (optional)
            j = int(np.argmin(dists2))
            nn_indices[i] = idxs[j]
            nn_dists2[i] = dists2[j]

    scan_in = scanned_pcd[mask]
    scan_out = scanned_pcd[~mask]

    # Keep only nearest neighbor info for the inliers
    nn_indices = nn_indices[mask]
    nn_dists2 = nn_dists2[mask]

    return scan_in, scan_out, mask, nn_indices, nn_dists2


scan_in, scan_out, mask_around_part, nn_idx, nn_d2 = scanned_points_in_part_radius(
    np.asarray(pcd_target.points), np.asarray(actual_pcd.points), r=0.005
)

scan_in_pcd = o3d.geometry.PointCloud()
scan_in_pcd.points = o3d.utility.Vector3dVector(scan_in)
scan_in_pcd.paint_uniform_color([1, 0, 1])

o3d.visualization.draw_geometries(
    [pcd_target, scan_in_pcd], window_name="Scanned Points in Target Part Range"
)


actual_pcd.scale(1000, np.array([0, 0, 0]))
points_target_in_scan = np.asarray(actual_pcd.points)[mask_around_part]
pcd_target_in_scan = o3d.geometry.PointCloud()
pcd_target_in_scan.points = o3d.utility.Vector3dVector(points_target_in_scan)
dists = pcd_target_in_scan.compute_nearest_neighbor_distance()
avg_d = np.mean(dists)
print(f"New Scale:{avg_d:.4f}")
o3d.visualization.draw_geometries(
    [pcd_target, pcd_target_in_scan], window_name="Scanned Points in Target Part Range"
)

# ************************
original_points = np.asarray(pcd_target_in_scan.points)
original_indices = np.arange(len(original_points))
# ----------------  initialize parameters for plane extraction ----------------
plane_indices_list = (
    []
)  # for each plane, the indices of its points in the original point cloud
plane_colors = []  # for each plane, its color
plane_models = []  # for each plane, its model (ax+by+cz+d)
plane_normals = []  # for each plane, its normal vector

rest_pcd = copy.deepcopy(pcd_target_in_scan)


# ---------------- Plan Segmentation ----------------
# ***************************************************
for i in range(max_planes):
    print(f"loop{i}")

    if len(rest_pcd.points) < min_remaining_points:
        print("Not enough points left to extract more planes.")
        break

    plane_model, inliers = rest_pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
    )

    if len(inliers) < min_points_per_plane:
        print(f"Plane has only {len(inliers)} points, skipping.")
        break

    # save the indices of current plane's points in the original point cloud
    current_points = np.asarray(rest_pcd.points)
    current_indices = np.arange(len(current_points))
    original_idx = original_indices[inliers]
    plane_indices_list.append(original_idx)

    # ***************************************************
    # save the plane model and its normal
    plane_models.append(plane_model)
    normal_vector = np.asarray(plane_model[0:3])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # normal_vector = correct_normal_direction_by_density(pcd, original_idx, normal_vector)
    plane_normals.append(normal_vector)

    # color
    color = [random.random(), random.random(), random.random()]
    plane_colors.append(color)

    # update the rest point cloud
    rest_pcd = rest_pcd.select_by_index(inliers, invert=True)
    original_indices = np.delete(original_indices, inliers)


def _normalize_plane(a, b, c, d):
    """normalize the plane equation ax+by+cz+d=0 to ||[a,b,c]|| = 1,return (n_hat, d_hat)"""
    n = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("Invalid plane normal with zero length.")
    n_hat = n / norm
    d_hat = d / norm
    return n_hat, d_hat


def _angle_deg(n1, n2):
    """The angle between the normal vectors of the two units (in degrees)."""
    cosv = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return np.degrees(np.arccos(cosv))


def _refit_plane_from_points(points_xyz):
    """
    Fit a plane to a set of points using SVD.
    Returns (plane_model[a,b,c,d], unit_normal)
    """
    P = np.asarray(points_xyz, dtype=float)
    if len(P) < 3:
        raise ValueError("Need at least 3 points to fit a plane.")
    centroid = P.mean(axis=0)
    Q = P - centroid
    # SVD: Normal corresponding to the smallest eigenvector
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    normal = vt[-1, :]
    normal /= np.linalg.norm(normal)
    d = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], d], dtype=float), normal


def merge_coplanar_planes(
    plane_models,  # [ [a,b,c,d], ... ]
    plane_normals,  # [ unit n_i, ... ]
    plane_indices_list,  # [ idx_list_i, ... ]
    all_points_xyz,  # np.asarray(pcd.points)
    angle_thresh_deg=5.0,  # Normal angle threshold (degrees)
    offset_thresh=1.0,  # Same direction |d1 - d2| threshold (unit: point cloud coordinates)
):
    """
    return:
      new_plane_models, new_normals, new_indices_list
    """

    planes = []
    for model, n_unit, idxs in zip(plane_models, plane_normals, plane_indices_list):
        a, b, c, d = model
        n_hat, d_hat = _normalize_plane(a, b, c, d)
        # If opposite to the recorded unit normal, flip (n, d) simultaneously.
        if np.dot(n_hat, n_unit) < 0:
            n_hat = -n_hat
            d_hat = -d_hat
        planes.append(
            {
                "n": n_hat,  # norm normal vector
                "d": d_hat,  # offset
                "idxs": np.asarray(idxs, dtype=int),
            }
        )

    # Merge each pair that can be merged until no further merging is possible.
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

                # ensure the normals are pointing in the same direction
                if np.dot(n1, n2) < 0:
                    n2_cmp = -n2
                    d2_cmp = -d2
                else:
                    n2_cmp = n2
                    d2_cmp = d2

                angle = _angle_deg(n1, n2_cmp)
                offset_diff = abs(d1 - d2_cmp)

                if angle <= angle_thresh_deg and offset_diff <= offset_thresh:
                    merged_pair = (i, j)
                    break
            if merged_pair is not None:
                break

        if merged_pair is not None:
            i, j = merged_pair
            # merge the indices of the two planes
            idxs_merged = np.unique(
                np.concatenate([planes[i]["idxs"], planes[j]["idxs"]], axis=0)
            )

            # Refit the plane using the merged points
            pts = all_points_xyz[idxs_merged]
            model_new, n_new = _refit_plane_from_points(pts)
            # Normalization
            n_hat, d_hat = _normalize_plane(*model_new)

            # Ensure the normal is pointing in the same direction as before
            if np.dot(n_hat, planes[i]["n"]) < 0:
                n_hat = -n_hat
                d_hat = -d_hat
                model_new = np.array([n_hat[0], n_hat[1], n_hat[2], d_hat])
            # Construct the new plane, replace i, and delete j
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


all_points_xyz = np.asarray(pcd_target_in_scan.points)
plane_models, plane_normals, plane_indices_list = merge_coplanar_planes(
    plane_models,
    plane_normals,
    plane_indices_list,
    all_points_xyz,
    angle_thresh_deg=plane_angle_thresh,
    offset_thresh=2.0,
)
print(f"Number of planes after merging: {len(plane_models)}")
for k, (m, n, idxs) in enumerate(zip(plane_models, plane_normals, plane_indices_list)):
    print(f"[{k}] model(a,b,c,d)= {m}, normal= {n}, num_points= {len(idxs)}")


# ----------------  Visualization ----------------

colored_pcd = copy.deepcopy(pcd_target_in_scan)
colors = np.ones((len(original_points), 3)) * [0.5, 0.5, 0.5]

for indices, color in zip(plane_indices_list, plane_colors):
    colors[indices] = color

colored_pcd.colors = o3d.utility.Vector3dVector(colors)

actual_pcd.paint_uniform_color([0.7, 0.7, 0.7])
o3d.visualization.draw_geometries(
    [colored_pcd.translate([0, 0, 0.1]), actual_pcd],
    window_name="Plane segmentation result and external normal",
)


# *****


def is_parallel(v1, v2, angle_thresh_deg=5):
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = degrees(acos(abs(cos_theta)))
    return angle <= angle_thresh_deg


unclustered = set(range(len(plane_normals)))
parallel_groups = []

while unclustered:
    idx = unclustered.pop()
    ref_normal = plane_normals[idx]

    current_group = [idx]

    to_remove = []
    for other in unclustered:
        if is_parallel(ref_normal, plane_normals[other], plane_angle_thresh):
            current_group.append(other)
            to_remove.append(other)

    for i in to_remove:
        unclustered.remove(i)

    parallel_groups.append(current_group)


colored_pcd = copy.deepcopy(pcd_target_in_scan)
colors = np.ones((len(pcd_target_in_scan.points), 3)) * [0.5, 0.5, 0.5]

group_colors = [
    [random.random(), random.random(), random.random()] for _ in parallel_groups
]

for group_idx, group in enumerate(parallel_groups):
    color = group_colors[group_idx]
    for plane_idx in group:
        point_indices = plane_indices_list[plane_idx]
        colors[point_indices] = color

colored_pcd.colors = o3d.utility.Vector3dVector(colors)

if no_image == False:
    if show_planes_parallel_clustering == True:
        o3d.visualization.draw_geometries(
            [colored_pcd, actual_pcd],
            window_name="Planeclustering result and external normal",
        )

############################################### Plane pair


def is_opposite_direction(idx_i, idx_j):
    pn1 = np.asarray(plane_normals[idx_i])
    pn2 = np.asarray(plane_normals[idx_j])

    point_indices_i = plane_indices_list[idx_i]
    point_indices_j = plane_indices_list[idx_j]
    plane_points_i = np.asarray(
        pcd_target_in_scan.select_by_index(point_indices_i).points
    )
    plane_points_j = np.asarray(
        pcd_target_in_scan.select_by_index(point_indices_j).points
    )
    pc1 = np.mean(plane_points_i, axis=0)
    pc2 = np.mean(plane_points_j, axis=0)

    c1c2 = pc2 - pc1

    dot0 = np.dot(pn1, pn2)
    dot1 = np.dot(c1c2, pn1)  # external normal :-
    dot2 = np.dot(c1c2, pn2)  # external normal :+

    if dot0 < 0 and dot1 < 0 and dot2 > 0:  # same direction
        return True
    else:
        return False


paired_planes = []  # all paired

for group in parallel_groups:
    n = len(group)
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = group[i]
            idx_j = group[j]

            n1 = plane_normals[idx_i]
            n2 = plane_normals[idx_j]

            # if is_opposite_direction(idx_i, idx_j):
            paired_planes.append((idx_i, idx_j))

print(
    f"\n\n=======================================\n========== Paired planes: {len(paired_planes)} ==========\n======================================="
)


for count, (i, j) in enumerate(paired_planes):
    colors = np.ones((len(pcd_target_in_scan.points), 3)) * [0.6, 0.6, 0.6]

    # Specify colors for the current pairing
    color = [random.random(), random.random(), random.random()]
    for idx in plane_indices_list[i]:
        colors[idx] = color
    for idx in plane_indices_list[j]:
        colors[idx] = color

    paired_pcd = copy.deepcopy(pcd_target_in_scan)
    paired_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Show pair：{i} ↔ {j}")

    if no_image == False:
        if show_plane_pairs == True:
            o3d.visualization.draw_geometries(
                [paired_pcd],
                window_name=f"Pair {count + 1}: Plane {i} & Plane {j}",
                width=800,
                height=600,
            )

# ---------------Find center plane---------------


counter = 0
for iii in range(len(paired_planes)):
    counter += 1
    print(
        f"\n\n----------------------------------------\n-------- Processing pair: {counter}/{len(paired_planes)} --------\n----------------------------------------"
    )

    mmm, nnn = paired_planes[iii]
    plane_i_points = np.asarray(
        pcd_target_in_scan.select_by_index(plane_indices_list[mmm]).points
    )
    plane_j_points = np.asarray(
        pcd_target_in_scan.select_by_index(plane_indices_list[nnn]).points
    )
    center_i = np.mean(plane_i_points, axis=0)
    center_j = np.mean(plane_j_points, axis=0)

    dist_plane = abs(np.dot(center_i - center_j, plane_normals[mmm]))

    print(
        f"Plane pair distance: {dist_plane:.3f},open:{(f_pg - 2 * w_pg)},close:{g_pg}"
    )
    obb = pcd_target_in_scan.get_oriented_bounding_box()
    obb_extent = max(obb.extent)
    print(f"OBB extent:{obb_extent}")

    if dist_plane < g_pg:
        print(
            "\n####################\nPlane pair distance is too small, skip\n####################\n"
        )
        continue
    elif dist_plane > (f_pg - 2 * w_pg):
        print(
            "\n####################\nPlane pair distance is too far, skip\n####################\n"
        )
        continue

    center_ij = (center_i + center_j) / 2

    dist_dir_i = np.dot(center_ij - center_i, plane_normals[mmm])
    dist_dir_i = -1.0 if dist_dir_i > 0 else 1.0
    dist_dir_j = np.dot(center_ij - center_j, plane_normals[nnn])
    dist_dir_j = -1.0 if dist_dir_i > 0 else 1.0

    dist_i = abs(np.dot((center_ij - center_i), plane_normals[mmm]))
    dist_j = abs(np.dot((center_ij - center_j), plane_normals[nnn]))

    # project_i_dir = (center_ij - center_i) / np.linalg.norm(center_ij - center_i)
    # project_j_dir = (center_ij - center_j) / np.linalg.norm(center_ij - center_j)

    projected_i_points = plane_i_points - dist_dir_i * np.outer(
        dist_i, plane_normals[mmm]
    )

    projected_j_points = plane_j_points - dist_dir_j * np.outer(
        dist_j, plane_normals[nnn]
    )

    pcd_proj_i = o3d.geometry.PointCloud()
    pcd_proj_i.points = o3d.utility.Vector3dVector(projected_i_points)
    pcd_proj_i.paint_uniform_color([0, 1, 0])

    pcd_proj_j = o3d.geometry.PointCloud()
    pcd_proj_j.points = o3d.utility.Vector3dVector(projected_j_points)
    pcd_proj_j.paint_uniform_color([1, 0, 0])

    pcd_orig_i = pcd_target_in_scan.select_by_index(plane_indices_list[mmm])
    pcd_orig_j = pcd_target_in_scan.select_by_index(plane_indices_list[nnn])

    pcd_orig_i.paint_uniform_color([0.85, 0.85, 0.85])
    pcd_orig_j.paint_uniform_color([0.85, 0.85, 0.85])

    if no_image == False:
        if show_plane_pair_and_proj_in_pcd == True:
            o3d.visualization.draw_geometries(
                [
                    actual_pcd.translate([0, 0.001, 0]),
                    pcd_orig_i,
                    pcd_orig_j,
                    pcd_proj_i,
                    pcd_proj_j,
                ],
                window_name="Plane Projection",
                width=800,
                height=600,
            )

    if no_skip == False:
        if input("Skip this pair? (y/n)") == "y":
            continue

    # **************************** Plane 1: Project planes and find overlap region ****************************

    def extract_overlap_region(proj_A, proj_B, threshold=0.001, remove=False):
        """
        Extract overlapping areas from two projected point clouds and return the merged overlapping point cloud.
        """

        dA = np.asarray(proj_A.compute_nearest_neighbor_distance())
        dB = np.asarray(proj_B.compute_nearest_neighbor_distance())
        print("median spacing A/B:", np.median(dA), np.median(dB))

        threshold = 1.2 * max(np.median(dA), np.median(dB))
        # Contrust KDTree
        kdtree_B = o3d.geometry.KDTreeFlann(proj_B)
        kdtree_A = o3d.geometry.KDTreeFlann(proj_A)

        points_A = np.asarray(proj_A.points)
        points_B = np.asarray(proj_B.points)

        # Points in A that have a neighbor in B within the threshold distance
        matched_A = []
        dismatched_A = []
        for p in points_A:
            [_, idx, _] = kdtree_B.search_radius_vector_3d(p, threshold)
            if len(idx) > 0:
                matched_A.append(p)
            else:
                dismatched_A.append(p)

        if remove == True:
            pcd_remove_overlap = o3d.geometry.PointCloud()
            pcd_remove_overlap.points = o3d.utility.Vector3dVector(dismatched_A)
            pcd_remove_overlap.paint_uniform_color([1, 0, 0])

            return pcd_remove_overlap
        else:
            # Points in B that have a neighbor in A within the threshold distance
            matched_B = []
            for p in points_B:
                [_, idx, _] = kdtree_A.search_radius_vector_3d(p, threshold)
                if len(idx) > 0:
                    matched_B.append(p)

            A_keep = np.array(matched_A, dtype=float).reshape(-1, 3)
            B_keep = np.array(matched_B, dtype=float).reshape(-1, 3)
            if A_keep.size == 0 and B_keep.size == 0:
                print(
                    "\n############################\nThere is no intersection between this pair of planes.\n############################\n"
                )
                return None
            else:
                overlap_points = (
                    A_keep
                    if B_keep.size == 0
                    else (B_keep if A_keep.size == 0 else np.vstack([A_keep, B_keep]))
                )

            if len(overlap_points) < 200:
                print(
                    "\n############################\nThere are less than 200 points here.\n############################\n"
                )
                return None

            pcd_overlap = o3d.geometry.PointCloud()
            pcd_overlap.points = o3d.utility.Vector3dVector(overlap_points)
            pcd_overlap.paint_uniform_color([0, 1, 0])

            return pcd_overlap

    overlap_pcd_unfilter = extract_overlap_region(
        pcd_proj_i, pcd_proj_j, threshold=0.001
    )
    if overlap_pcd_unfilter is None:
        continue

    overlap_pcd, ind_p1 = filter_by_normal_orientation(
        overlap_pcd_unfilter, plane_normals[mmm]
    )
    overlap_pcd, ind_p1 = remove_pcd_outlier_statistical(overlap_pcd_unfilter)
    projected_points_p1 = np.asarray(overlap_pcd.points)

    colors = np.ones((len(overlap_pcd_unfilter.points), 3)) * [1, 1, 0]
    colors[ind_p1, :] = [0, 1, 0]
    overlap_pcd_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p1 == True:
            # o3d.visualization.draw_geometries([overlap_pcd_unfilter.translate([0,0,0.00001]),pcd_orig_i,pcd_orig_j],window_name="Pair of Planes and Their Overlap Region")
            o3d.visualization.draw_geometries(
                [overlap_pcd_unfilter.translate([0, 0, 0.00001])],
                window_name="Pair of Planes and Their Overlap Region",
            )

    # **************************** Plane 2: Find points between planes ****************************
    def project_points_to_plane(points, plane_point, plane_normal):
        v = points - plane_point
        d = np.dot(v, plane_normal)
        return points - np.outer(d, plane_normal)

    def select_points_between_planes(
        pcd, center_i, center_j, plane_normal, margin=0.0015, include_planes=True
    ):
        """
        Filter out points between two planes in the complete point cloud
        """

        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
        elif isinstance(pcd, np.ndarray):
            points = pcd
        # center_i = plane_i_pts.mean(axis=0)
        # center_j = plane_j_pts.mean(axis=0)

        # Planar distance vector (direction must be consistent with the normal vector)
        dist_vec = center_j - center_i
        dist_vec /= np.linalg.norm(dist_vec)
        # Project each point to the normal vector, and get the distance from the two planes
        d_i = np.dot(points - center_i, plane_normal)
        d_j = np.dot(points - center_j, plane_normal)

        # Determine whether a point lies between two planes (allow for a margin)
        if include_planes:
            mask = (d_i * d_j <= 0) | (np.abs(d_i) <= margin) | (np.abs(d_j) <= margin)
        else:
            mask = (d_i * d_j < 0) & (np.abs(d_i) > margin) & (np.abs(d_j) > margin)

        points_between = points[mask]
        points_beside = points[~mask]
        return points_between, points_beside

    points_between_p2, points_beside = select_points_between_planes(
        actual_pcd,
        center_i,
        center_j,
        plane_normals[mmm],
        margin=margin_points_between_planes,
    )

    projected_points_p2 = project_points_to_plane(
        points_between_p2, center_ij, plane_normals[mmm]
    )

    proj_pcd_p2_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p2_unfilter.points = o3d.utility.Vector3dVector(projected_points_p2)
    proj_pcd_p2_unfilter.paint_uniform_color([1, 0, 0])

    proj_pcd_p2, ind_p2 = remove_pcd_outlier_statistical(proj_pcd_p2_unfilter, 20, 1.0)
    projected_points_p2 = np.asarray(proj_pcd_p2.points)

    colors = np.ones((len(proj_pcd_p2_unfilter.points), 3)) * [1, 1, 0]
    colors[ind_p2, :] = [1, 0, 0]
    proj_pcd_p2_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p2 == True:
            # o3d.visualization.draw_geometries([
            #     pcd_orig_j, pcd_orig_i, proj_pcd_p2_unfilter
            #     ,
            # ],window_name="Pair of Planes and Projected Points Between Them")
            o3d.visualization.draw_geometries(
                [
                    proj_pcd_p2_unfilter,
                ],
                window_name="Pair of Planes and Projected Points Between Them",
            )

    # **************************** Plane 3: find within finger width collision area ****************************

    center_i_p3 = center_i + (a_pg + w_pg + v_pg) * (plane_normals[mmm]) * dist_dir_i
    center_j_p3 = center_j + (a_pg + w_pg + v_pg) * (plane_normals[nnn]) * dist_dir_j
    # center_i_p3 = center_ij + (0.02) * (plane_normals[mmm]) * dist_dir_i
    # center_j_p3 = center_ij + (0.02) * (plane_normals[nnn]) * dist_dir_j

    points_between_p3_i, points_beside = select_points_between_planes(
        points_beside,
        center_i,
        center_i_p3,
        plane_normals[mmm],
        margin=margin_points_between_planes,
    )
    points_between_p3_j, points_beside = select_points_between_planes(
        points_beside,
        center_j,
        center_j_p3,
        plane_normals[nnn],
        margin=margin_points_between_planes,
    )
    points_between_p3 = np.vstack((points_between_p3_i, points_between_p3_j))

    projected_points_p3 = project_points_to_plane(
        points_between_p3, center_ij, plane_normals[mmm]
    )

    proj_pcd_p3_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p3_unfilter.points = o3d.utility.Vector3dVector(projected_points_p3)
    # proj_pcd_p3_unfilter.paint_uniform_color([0, 0, 1])

    proj_pcd_p3, ind_p3 = remove_pcd_outlier_statistical(proj_pcd_p3_unfilter, 50, 2.0)
    projected_points_p3 = np.asarray(proj_pcd_p3.points)

    colors = np.ones((len(proj_pcd_p3_unfilter.points), 3)) * [1, 1, 0]
    colors[ind_p3, :] = [0, 0, 1]
    proj_pcd_p3_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p3 == True:
            o3d.visualization.draw_geometries(
                [overlap_pcd, proj_pcd_p3_unfilter.translate([0, 0, 0.0001])],
                window_name="Initial TCP & Finger Collision Area (P3)",
            )

    # **************************** Plane 4: find outside finger width collision area ****************************

    center_i_p4 = center_ij + (y_pg / 2) * (plane_normals[mmm]) * dist_dir_i
    center_j_p4 = center_ij + (y_pg / 2) * (plane_normals[nnn]) * dist_dir_j

    points_between_p4_i, points_beside = select_points_between_planes(
        points_beside,
        center_i_p3,
        center_i_p4,
        plane_normals[mmm],
        margin=margin_points_between_planes,
    )
    points_between_p4_j, points_beside = select_points_between_planes(
        points_beside,
        center_j_p3,
        center_j_p4,
        plane_normals[nnn],
        margin=margin_points_between_planes,
    )
    points_between_p4 = np.vstack((points_between_p4_i, points_between_p4_j))

    projected_points_p4 = project_points_to_plane(
        points_between_p4, center_ij, plane_normals[mmm]
    )

    proj_pcd_p4_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p4_unfilter.points = o3d.utility.Vector3dVector(projected_points_p4)
    # proj_pcd_p4_unfilter.paint_uniform_color([0, 0, 1])

    proj_pcd_p4, ind_p4 = remove_pcd_outlier_statistical(proj_pcd_p4_unfilter, 50, 3.0)
    projected_points_p4 = np.asarray(proj_pcd_p4.points)

    colors = np.ones((len(proj_pcd_p4_unfilter.points), 3)) * [1, 1, 0]
    colors[ind_p4, :] = [0, 0.5, 1]
    proj_pcd_p4_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p4 == True:
            # o3d.visualization.draw_geometries([overlap_pcd,proj_pcd_p3_unfilter.translate([0,0,-0.0001]), proj_pcd_p4_unfilter.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3+P4)")
            o3d.visualization.draw_geometries(
                [overlap_pcd, proj_pcd_p4_unfilter.translate([0, 0, 0.0001])],
                window_name="Initial TCP & Finger Collision Area (P3+P4)",
            )

    ##**************************** Plane 5: find beside collision area ****************************

    center_i_p5 = center_ij + ((rd + rj) / 2) * (plane_normals[mmm]) * dist_dir_i
    center_j_p5 = center_ij + ((rd + rj) / 2) * (plane_normals[nnn]) * dist_dir_j

    points_between_p5_i, points_beside = select_points_between_planes(
        points_beside,
        center_i_p4,
        center_i_p5,
        plane_normals[mmm],
        margin=margin_points_between_planes,
    )
    points_between_p5_j, points_beside = select_points_between_planes(
        points_beside,
        center_j_p4,
        center_j_p5,
        plane_normals[nnn],
        margin=margin_points_between_planes,
    )
    points_between_p5 = np.vstack((points_between_p5_i, points_between_p5_j))

    projected_points_p5 = project_points_to_plane(
        points_between_p5, center_ij, plane_normals[mmm]
    )

    proj_pcd_p5_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p5_unfilter.points = o3d.utility.Vector3dVector(projected_points_p5)
    proj_pcd_p5_unfilter.paint_uniform_color([0, 1, 1])

    proj_pcd_p5, ind_p5 = remove_pcd_outlier_dbscan(proj_pcd_p5_unfilter)
    # proj_pcd_p5.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    projecter_points_p5 = np.asarray(proj_pcd_p5.points)

    colors = np.ones((len(proj_pcd_p5_unfilter.points), 3)) * [1, 1, 0]
    colors[ind_p5, :] = [0, 1, 1]
    proj_pcd_p5_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p5 == True:
            # o3d.visualization.draw_geometries([overlap_pcd, proj_pcd_p3.translate([0,0,0.0001]), proj_pcd_p4.translate([0,0,0.0002]), proj_pcd_p5_unfilter.translate([0,0,-0.0001])],window_name="Initial TCP & Finger Collision Area (P3+P4) & Robot Collision Area (P5)")
            o3d.visualization.draw_geometries(
                [overlap_pcd, proj_pcd_p5_unfilter.translate([0, 0, -0.0001])],
                window_name="Initial TCP & Finger Collision Area (P3+P4) & Robot Collision Area (P5)",
            )

    # ********************** SHOW P1 P2 P3 P4 P5 WITH PCD ****************

    pcd_between_p22 = o3d.geometry.PointCloud()
    pcd_between_p22.points = o3d.utility.Vector3dVector(points_between_p2)
    pcd_between_p22.paint_uniform_color([1, 0, 0])

    pcd_between_p33 = o3d.geometry.PointCloud()
    pcd_between_p33.points = o3d.utility.Vector3dVector(points_between_p3)
    pcd_between_p33.paint_uniform_color([0, 0, 1])

    pcd_between_p44 = o3d.geometry.PointCloud()
    pcd_between_p44.points = o3d.utility.Vector3dVector(points_between_p4)
    pcd_between_p44.paint_uniform_color([0, 0.5, 1])

    pcd_beside_p5 = o3d.geometry.PointCloud()
    pcd_beside_p5.points = o3d.utility.Vector3dVector(points_beside)
    pcd_beside_p5.paint_uniform_color([0, 1, 1])
    if no_image == False:
        if show_P2345_in_pcd == True:
            o3d.visualization.draw_geometries(
                [
                    pcd_between_p22,
                    pcd_between_p33.translate([0, 0, 0.0001]),
                    pcd_between_p44.translate([0, 0, 0.0002]),
                    pcd_beside_p5.translate([0, 0, -0.0001]),
                ],
                window_name="PCD [P2 P3 P4 P5] in 3D",
            )

    # *Show projected points on P1 P2 P3 P4 P5 with assemble PCD in 3D

    actual_pcd.paint_uniform_color([0.85, 0.85, 0.85])

    if no_image == False:
        if show_each_P_in_pcd == True:
            o3d.visualization.draw_geometries(
                [actual_pcd, overlap_pcd.translate([0, 0, 0.0001])],
                window_name="PCD+P1",
            )
            o3d.visualization.draw_geometries(
                [actual_pcd, proj_pcd_p2.translate([0, 0, 0.0001])],
                window_name="PCD+P2",
            )
            o3d.visualization.draw_geometries(
                [actual_pcd, proj_pcd_p3.translate([0, 0, -0.0001])],
                window_name="PCD+P3",
            )
            o3d.visualization.draw_geometries(
                [actual_pcd, proj_pcd_p4.translate([0, 0, -0.0001])],
                window_name="PCD+P4",
            )
            o3d.visualization.draw_geometries(
                [actual_pcd, proj_pcd_p5.translate([0, 0, -0.0001])],
                window_name="PCD+P5",
            )

    # **************************** P2: Find contours ****************************

    def auto_img_scale(pcd, target_size=512):
        points = np.asarray(pcd.points)
        # Calculate the width and height of the original point cloud in the 2D main plane.
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        ranges = max_vals - min_vals

        # Select the larger side as the reference point and uniformly scale it to target_size
        scale = target_size / np.max(ranges)
        return 1

    # -----------------------chat-cv2------------------------

    def extract_and_visualize_contour_segments_with_normals(
        pcd, scale=1500, approx_eps_ratio=0.01
    ):
        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
        elif isinstance(pcd, np.ndarray):
            print("Find contours Error: Input is not a PointCloud object.")
            return

        # PCA main direction (dir1, dir2) for building local plane.
        pca = PCA(n_components=3)
        pca.fit(points)
        dir1, dir2 = pca.components_[0], pca.components_[1]
        center = pca.mean_

        # project points to the main plane (2D)
        points = np.dot(points - center, np.vstack([dir1, dir2]).T)

        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)

        # transform points to image coordinates
        img_scale = auto_img_scale(pcd)
        points_img = np.int32((points - min_pt) * img_scale) + contour_image_padding
        img_size = ((max_pt - min_pt) * img_scale).astype(
            int
        ) + 2 * contour_image_padding

        # Create a blank image and draw points
        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for pt in points_img:
            cv2.circle(img, tuple(pt), 1, 255, -1)

            # ---- Estimate Typical Pixel Spacing of Points (for Kernel Selection) ----

        # Does not rely on third-party libraries: roughly estimates spacing from point image
        # Approach: downsample the grid or use sparsity approximation; here we use an approximate morphological distance method
        ys, xs = np.where(img > 0)
        if len(xs) >= 2:
            # Use a small window to estimate the nearest pixel distance (simplified estimation)
            # Alternatively, use scipy.spatial.cKDTree to calculate median pixel spacing via nearest neighbors
            sample = np.random.choice(len(xs), size=min(5000, len(xs)), replace=False)
            pts = np.stack([xs[sample], ys[sample]], axis=1).astype(np.int32)
            # Use small-radius erosion to check if points disconnect, approximately inferring spacing (conservative value)
            # Simplified: use a constant fallback value
            px_gap = 3
        else:
            px_gap = 3

        # ---- Morphology: Close operation first, then open operation ----
        # Kernal size is linked to point spacing, closed operations fill gaps, and open operations remove burrs.
        k = max(
            3, int(round(px_gap * 2))
        )  # Closed operation core (the larger the better)
        k_open = max(
            3, int(round(px_gap * 0.8))
        )  # Calculation core (light noise reduction)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # ---- Fill holes (make sure you get the true outer contour)----
        h, w = mask.shape
        ff = mask.copy()
        ff = cv2.copyMakeBorder(ff, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(ff, None, (0, 0), 255)  # Flooding from outside the boundary
        ff = ff[1:-1, 1:-1]
        holes = cv2.bitwise_not(ff) & cv2.bitwise_not(mask)  # External area
        filled = cv2.bitwise_or(mask, cv2.bitwise_not(holes))

        # ---- Remove small connected regions (to prevent isolated points from affecting the outer contour)----
        num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        min_area_px = (k * k) * 2  # Area threshold: related to the kernel
        clean = np.zeros_like(filled)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                clean[labels == i] = 255

        # findContours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Drawing various contour processing results
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(img_contours, [approx], 0, (0, 255, 255), 2)

        # Display contours and advanced processing results
        plt.figure(figsize=(6, 6))  # Contour P2
        plt.imshow(img_contours)
        plt.title("Contours: Plane 2")
        plt.axis("off")
        if no_image == False and show_plt_contour_P2_2d == True:
            plt.show()
        else:
            plt.close()

        # ---- Convert the contour back to the original coordinate space. ---- #
        contours_real = []  # Each contour in the 2D projection coordinate system
        polygons_2d = []  # shapely Polygon list
        # contour_points_list = []
        linesets = []

        for cnt in contours:
            # Approximate polygon contour
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

            # Converting back from the image coordinate system to the projected two-dimensional coordinate system
            points_2d_back = (
                approx.astype(np.float32) - contour_image_padding
            ) / img_scale + min_pt

            # Mapping from two-dimensional coordinates back to the original three-dimensional space
            points_3d = np.dot(points_2d_back, np.vstack([dir1, dir2])) + center

            line_segments_2d, line_normals_2d = [], []
            line_segments_3d, line_indices, line_colors = [], [], []

            for i in range(len(points_2d_back)):
                pt1_2d = points_2d_back[i]
                pt2_2d = points_2d_back[(i + 1) % len(points_2d_back)]  # closed

                # Line segment direction and normal
                vec = pt2_2d - pt1_2d
                length = np.linalg.norm(vec)
                if length == 0:
                    continue
                direction = vec / length
                normal_2d = np.array([-direction[1], direction[0]])

                line_segments_2d.append([pt1_2d, pt2_2d])
                line_normals_2d.append(normal_2d)

                # Projection back to 3D space
                pt1_3d = center + pt1_2d[0] * dir1 + pt1_2d[1] * dir2
                pt2_3d = center + pt2_2d[0] * dir1 + pt2_2d[1] * dir2

                # Line segment added to LineSet
                idx = len(line_segments_3d)
                line_segments_3d.extend([pt1_3d, pt2_3d])
                line_indices.append([idx, idx + 1])
                color = plt.cm.hsv(i / len(points_2d_back))[:3]
                line_colors.append(color)

            # Construct all line segments
            if len(line_indices) == 0:
                print(
                    "No visible line segments (possibly due to the outline being too small or overly simplified)"
                )
                return

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(
                np.asarray(line_segments_3d, dtype=float)
            )
            line_set.lines = o3d.utility.Vector2iVector(
                np.asarray(line_indices, dtype=np.int32)
            )
            line_set.colors = o3d.utility.Vector3dVector(
                np.asarray(line_colors, dtype=float)
            )

        if no_image == False:
            if show_P2_contour_3d == True:
                o3d.visualization.draw_geometries(
                    [pcd, line_set],
                    window_name="P2 Contour Lines + Normals",
                    width=1280,
                    height=800,
                )

        return (
            line_segments_2d,
            line_normals_2d,
            dir1,
            dir2,
            center,
            [line_segments_3d, line_indices],
        )

    contour_segments_2d_p2 = []
    contour_normals_2d_p2 = []
    (
        contour_segments_2d_p2,
        contour_normals_2d_p2,
        dir1,
        dir2,
        center,
        contour_segments_3d_p2_para,
    ) = extract_and_visualize_contour_segments_with_normals(
        proj_pcd_p2, scale=1500, approx_eps_ratio=0.01
    )

    # **************************** Find and Show Initial TCP Box & Test Grid Point ****************************
    def generate_grid_by_spacing(
        segments_2d, normals_2d, depth=0.05, spacing_edge=0.005, spacing_normal=0.005
    ):
        """
        Generate equally spaced grid points within rectangles expanded along the normal direction of each line segment.

        Parameters:
            segments_2d: List of (pt1, pt2), start and end points of 2D line segments
            normals_2d: List of unit normal vectors, one for each line segment
            depth: Width of the grasping region along the normal direction (in meters)
            spacing_edge: Grid point spacing along the edge direction (in meters)
            spacing_normal: Grid point spacing along the normal direction (in meters)

        Returns:
            rectangles: Rectangles corresponding to each line segment (4 points each)
            all_grid_points: Generated points within each rectangle, List[np.ndarray]
        """
        rectangles = []
        all_grid_points = []

        eps = 1e-9

        for (pt1, pt2), n in zip(segments_2d, normals_2d):
            pt1 = np.array(pt1)
            pt2 = np.array(pt2)
            n = np.array(n) / np.linalg.norm(n)

            # Line segment direction and length
            dir_vec = pt2 - pt1
            seg_len = np.linalg.norm(dir_vec)
            dir_unit = dir_vec / seg_len

            # Number of steps in the direction of the decision
            num_w = int(np.floor((seg_len - eps) / spacing_edge) + 1)
            start_spacing_edge = (seg_len - (num_w - 1) * spacing_edge) / 2.0
            num_d = int(np.floor((depth - eps) / spacing_normal) + 1)
            start_spacing_normal = (depth - (num_d - 1) * spacing_normal) / 2.0
            if num_w < 1 or num_d < 1:
                continue

            # Construct the four points of a rectangle (counterclockwise)
            offset = -n * depth
            p1 = pt1 + offset
            p2 = pt2 + offset
            p3 = pt2
            p4 = pt1
            rectangles.append([p1, p2, p3, p4])

            # Generate grid points inside a rectangle
            grid_pts = []
            for i in range(num_w):
                for j in range(num_d):
                    alpha = i * spacing_edge + start_spacing_edge
                    beta = j * spacing_normal + start_spacing_normal
                    # alpha = i * spacing_edge
                    # beta = j * spacing_normal
                    pt = p1 + dir_unit * alpha + n * beta
                    grid_pts.append(pt)
            all_grid_points.append(np.array(grid_pts))

        return rectangles, all_grid_points

    def plot_segments_tcpbox_and_grids(segments_2d, rectangles, grid_points):
        """
        Plot on a 2D coordinate plane:
        - Original line segments (blue)
        - Rectangular region for each line segment (green dashed line)
        - Regular grid points inside the rectangles (red x markers)
        """

        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - plt_graphic_padding
        max_xy = all_pts.max(axis=0) + plt_graphic_padding

        fig, ax = plt.subplots(figsize=(9, 8))  # All TCP

        used_labels = set()  # Track added legend labels

        for (pt1, pt2), rect, grids in zip(segments_2d, rectangles, grid_points):
            # Egde
            lbl = "Edges of Plane2"
            if lbl not in used_labels:
                ax.plot(
                    [pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3, label=lbl
                )
                used_labels.add(lbl)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3)

            # TCP Box
            rect = np.array(rect + [rect[0]])
            lbl = "TCP Box"
            if lbl not in used_labels:
                ax.plot(rect[:, 0], rect[:, 1], "g--", linewidth=0.5, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(rect[:, 0], rect[:, 1], "g--", linewidth=0.5)

            # Grid Points
            grids = np.array(grids)
            lbl = "Test Grid Points"
            if lbl not in used_labels:
                ax.plot(grids[:, 0], grids[:, 1], "rx", markersize=4, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(grids[:, 0], grids[:, 1], "rx", markersize=4)

        ax.set_aspect("equal")
        ax.set_title("All TCP Boxes and all Test Grid Points")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        os.makedirs("Image", exist_ok=True)
        plt.savefig("Image\\possible_tcp_all.svg", format="svg", bbox_inches="tight")
        plt.show()

    tcp_box, test_grid_points = generate_grid_by_spacing(
        contour_segments_2d_p2,
        contour_normals_2d_p2,
        depth=b_pg + c_pg,
        spacing_edge=z_pg / 5,
        spacing_normal=b_pg / 5,
    )

    if no_image == False and show_plt_all_tcp_grids == True:
        plot_segments_tcpbox_and_grids(
            contour_segments_2d_p2, tcp_box, test_grid_points
        )

    # Show each TCP Boxes and it's test grid points
    def highlight_segment_rect_grid(segments_2d, rectangles, grid_points):
        """
        Always display all edges, highlighting only the rectangle and grid points corresponding to the current index.
        """
        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - plt_graphic_padding
        max_xy = all_pts.max(axis=0) + plt_graphic_padding

        for i in range(len(segments_2d)):
            fig, ax = plt.subplots(figsize=(8, 8))  # each TCP
            ax.set_title(
                f"TCP Box and Test Grid Points for Edges {i + 1}/{len(segments_2d)}"
            )

            lbl = "Edges of Plane2"
            used_labels = set()
            for j, (pt1, pt2) in enumerate(segments_2d):
                if j == i:
                    mid = (pt1 + pt2) / 2
                    vec_12 = pt2 - pt1
                    vec_12 = vec_12 / np.linalg.norm(vec_12)
                    normal_clockwise_90 = [vec_12[1], -vec_12[0]]
                    normal_clockwise_90 = normal_clockwise_90 / np.linalg.norm(
                        normal_clockwise_90
                    )

                    # tilt symbol
                    start_point_line = (
                        mid - normal_clockwise_90 * tilt_symbol_start_dist
                    )
                    end_point_line = (
                        start_point_line
                        + normal_clockwise_90 * tilt_symbol_handle_length
                    )
                    end_point_base1 = (
                        end_point_line
                        + vec_12 * tilt_symbol_finger_width_half
                        - normal_clockwise_90 * tilt_symbol_finger_width_half
                    )
                    end_point_base2 = (
                        end_point_line
                        - vec_12 * tilt_symbol_finger_width_half
                        + normal_clockwise_90 * tilt_symbol_finger_width_half
                    )
                    end_point_finger1 = (
                        end_point_base1
                        + normal_clockwise_90 * tilt_symbol_finger_end_length
                    )
                    end_point_finger2 = (
                        end_point_base2
                        + normal_clockwise_90 * tilt_symbol_finger_end_length
                    )

                    ax.plot(
                        [start_point_line[0], end_point_line[0]],
                        [start_point_line[1], end_point_line[1]],
                        "m",
                        linewidth=1.5,
                        label="Gripper Direction",
                    )
                    ax.plot(
                        [end_point_base1[0], end_point_base2[0]],
                        [end_point_base1[1], end_point_base2[1]],
                        "m",
                        linewidth=1.5,
                    )
                    ax.plot(
                        [end_point_base1[0], end_point_finger1[0]],
                        [end_point_base1[1], end_point_finger1[1]],
                        "m",
                        linewidth=1.5,
                    )
                    ax.plot(
                        [end_point_base2[0], end_point_finger2[0]],
                        [end_point_base2[1], end_point_finger2[1]],
                        "m",
                        linewidth=1.5,
                    )

                if lbl not in used_labels:
                    ax.plot(
                        [pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3, label=lbl
                    )
                    used_labels.add(lbl)
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3)

            # Current rectangle: green
            rect = np.array(rectangles[i] + [rectangles[i][0]])
            ax.plot(rect[:, 0], rect[:, 1], "g--", linewidth=0.5, label="TCP Box")

            # Current grid point: red
            grids = np.array(grid_points[i])
            if grid_points is not None:
                ax.plot(
                    grids[:, 0],
                    grids[:, 1],
                    "rx",
                    markersize=4,
                    label="Test Grid Points",
                )

            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            os.makedirs("Image", exist_ok=True)
            plt.savefig(
                f"Image\\possible_tcp_edge_{i + 1}.svg",
                format="svg",
                bbox_inches="tight",
            )
            plt.show()

    if no_image == False:
        if show_plt_TCP_each_edge == True:
            highlight_segment_rect_grid(
                contour_segments_2d_p2, tcp_box, test_grid_points
            )

    # Show Gripper Bounding Box
    def create_gripper_bounding_box(grid_points, segments_2d):
        all_shapes = []
        segment_directions = [pt2 - pt1 for pt1, pt2 in segments_2d]
        segment_middle_point = [(pt1 + pt2) / 2 for pt1, pt2 in segments_2d]

        for pts, seg_dir, mid in zip(
            grid_points, segment_directions, segment_middle_point
        ):
            seg_dir = seg_dir / np.linalg.norm(seg_dir)
            normal = np.array([-seg_dir[1], seg_dir[0]])

            segment_shapes = []

            for pt in pts:
                pt = np.array(pt)
                grid_edge_distance = np.dot(mid - pt, normal)

                rectangles = []

                # Safespace Finger front
                center1 = pt - normal * (x_pg + rj)
                p11 = center1 + seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                p12 = (
                    center1
                    + seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                    + normal * (x_pg + rj)
                )
                p13 = (
                    center1
                    - seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                    + normal * (x_pg + rj)
                )
                p14 = center1 - seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                rectangles.append([p11, p12, p13, p14])

                # Finger length
                center2 = pt
                p21 = center2 + seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                p22 = (
                    center2
                    + seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                    + normal * (b_pg + c_pg + rj)
                )
                p23 = (
                    center2
                    - seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                    + normal * (b_pg + c_pg + rj)
                )
                p24 = center2 - seg_dir * (e_pg + 2 * (i_pg + rj)) / 2
                rectangles.append([p21, p22, p23, p24])

                # Gripper Base
                center3 = center2 + normal * (b_pg + c_pg + rj)
                p31 = center3 + seg_dir * (j_pg + 2 * rj) / 2
                p32 = (
                    center3
                    + seg_dir * (j_pg + 2 * rj) / 2
                    + normal * (d_pg + t_pg + u_pg + rj)
                )
                p33 = (
                    center3
                    - seg_dir * (j_pg + 2 * rj) / 2
                    + normal * (d_pg + t_pg + u_pg + rj)
                )
                p34 = center3 - seg_dir * (j_pg + 2 * rj) / 2
                rectangles.append([p31, p32, p33, p34])

                # Robot Arm
                center4 = center3 + normal * (d_pg + t_pg + u_pg + rj)
                p41 = center4 + seg_dir * (rd + re + 2 * rj) / 2
                p42 = center4 - seg_dir * (rd + re + 2 * rj) / 2
                p43 = (
                    center4
                    - seg_dir * (rd + re + 2 * rj) / 2
                    + normal * (rc + rf + 2 * rj)
                )
                p44 = (
                    center4
                    + seg_dir * (rd + re + 2 * rj) / 2
                    + normal * (rc + rf + 2 * rj)
                )
                rectangles.append([p41, p42, p43, p44])

                # Gripper Area
                center5 = pt
                p51 = center5 + seg_dir * (z_pg - 2 * rj) / 2
                p52 = center5 + seg_dir * (z_pg - 2 * rj) / 2 + normal * (b_pg - 2 * rj)
                p53 = center5 - seg_dir * (z_pg - 2 * rj) / 2 + normal * (b_pg - 2 * rj)
                p54 = center5 - seg_dir * (z_pg - 2 * rj) / 2
                rectangles.append([p51, p52, p53, p54])

                # Robot Back Space
                center6 = center4 + normal * (rc + rf + 2 * rj)
                p61 = center6 + seg_dir * (rd + re + 2 * rj) / 2
                p62 = (
                    center6
                    + seg_dir * (rd + re + 2 * rj) / 2
                    + normal * (grid_edge_distance + x_pg + rj)
                )
                p63 = (
                    center6
                    - seg_dir * (rd + re + 2 * rj) / 2
                    + normal * (grid_edge_distance + x_pg + rj)
                )
                p64 = center6 - seg_dir * (rd + re + 2 * rj) / 2
                rectangles.append([p61, p62, p63, p64])

                segment_shapes.append({"point": pt, "rectangles": rectangles})

            all_shapes.append(segment_shapes)

        return all_shapes

    def show_gripper_bounding_box(segments_2d, tcp_box, shapes):
        all_pts = [pt for seg in segments_2d for pt in seg]
        bounds = np.array(all_pts)
        min_xy = bounds.min(axis=0) - plt_graphic_padding
        max_xy = bounds.max(axis=0) + plt_graphic_padding

        for i, segment_shape in enumerate(shapes):
            for j, shape in enumerate(segment_shape):
                fig, ax = plt.subplots(figsize=(8, 8))  # Bounding Boxes
                ax.set_title(f"Edge {i + 1}, Point {j + 1}: Bounding Boxes")

                used_labels = set()

                # Edges（Blue）
                lbl = "Edges of Plane2"
                for pt1, pt2 in segments_2d:
                    if lbl not in used_labels:
                        ax.plot(
                            [pt1[0], pt2[0]],
                            [pt1[1], pt2[1]],
                            "b-",
                            linewidth=1,
                            label=lbl,
                        )
                        used_labels.add(lbl)
                    else:
                        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=1)

                # TCP Box
                rect = np.array(tcp_box[i] + [tcp_box[i][0]])
                lbl = "TCP Box"
                if lbl not in used_labels:
                    ax.plot(rect[:, 0], rect[:, 1], "g--", linewidth=0.5, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(rect[:, 0], rect[:, 1], "g--", linewidth=0.5)

                # Current TCP
                pt = shape["point"]
                lbl = "Test Point"
                if lbl not in used_labels:
                    ax.plot(pt[0], pt[1], "ro", markersize=4, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(pt[0], pt[1], "ro", markersize=4)

                # Bounding Boxes
                colors = [
                    "red",
                    "purple",
                    "orange",
                    "deepskyblue",
                    "yellow",
                    "limegreen",
                ]
                Box_label = [
                    "Finger Clearence Box",
                    "Finger Box",
                    "Finger Base Box",
                    "Robot Arm Box",
                    "Gripper Area Box",
                    "Robot Back Clearence Box",
                ]
                for k, rect in enumerate(shape["rectangles"]):
                    poly = np.array(rect + [rect[0]])
                    lbl = Box_label[k]
                    if lbl not in used_labels:
                        ax.plot(
                            poly[:, 0],
                            poly[:, 1],
                            color=colors[k],
                            linewidth=1.5,
                            label=lbl,
                        )
                        used_labels.add(lbl)
                    else:
                        ax.plot(poly[:, 0], poly[:, 1], color=colors[k], linewidth=1.5)

                ax.set_xlim(min_xy[0], max_xy[0])
                ax.set_ylim(min_xy[1], max_xy[1])
                ax.set_aspect("equal")
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.show()

    points_and_gripper_bounding_box = create_gripper_bounding_box(
        test_grid_points, contour_segments_2d_p2
    )

    if no_image == False:
        if show_plt_bounding_boxes == True:
            show_gripper_bounding_box(
                contour_segments_2d_p2, tcp_box, points_and_gripper_bounding_box
            )

    # ************************ Project P134 to 2 (CV2) **********************

    # *******************************

    def get_plane_contour_polygon(pcd, dir1, dir2, center, plane_name=""):
        if pcd.is_empty():
            return Polygon()

        points = np.asarray(pcd.points)

        if points.size <= 50:
            return Polygon()

        # project points to the main plane (2D)
        points = np.dot(points - center, np.vstack([dir1, dir2]).T)

        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)

        # transform points to image coordinates
        img_scale = auto_img_scale(pcd)
        points_img = np.int32((points - min_pt) * img_scale) + contour_image_padding
        img_size = ((max_pt - min_pt) * img_scale).astype(
            int
        ) + 2 * contour_image_padding

        # Create a blank image and draw points
        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for pt in points_img:
            cv2.circle(img, tuple(pt), 1, 255, -1)

            # ---- Estimate Typical Pixel Spacing of Points (for Kernel Selection) ----

        # Does not rely on third-party libraries: roughly estimates spacing from point image
        # Approach: downsample the grid or use sparsity approximation; here we use an approximate morphological distance method
        ys, xs = np.where(img > 0)
        if len(xs) >= 2:
            # Use a small window to estimate the nearest pixel distance (simplified estimation)
            # Alternatively, use scipy.spatial.cKDTree to calculate median pixel spacing via nearest neighbors
            sample = np.random.choice(len(xs), size=min(5000, len(xs)), replace=False)
            pts = np.stack([xs[sample], ys[sample]], axis=1).astype(np.int32)
            # Use small-radius erosion to check if points disconnect, approximately inferring spacing (conservative value)
            # Simplified: use a constant fallback value
            px_gap = 3
        else:
            px_gap = 3

        # ---- Morphology: Close operation first, then open operation ----
        # Kernal size is linked to point spacing, closed operations fill gaps, and open operations remove burrs.
        k = max(
            3, int(round(px_gap * 2))
        )  # Closed operation core (the larger the better)
        k_open = max(
            3, int(round(px_gap * 0.8))
        )  # Calculation core (light noise reduction)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # ---- Fill holes (make sure you get the true outer contour)----
        h, w = mask.shape
        ff = mask.copy()
        ff = cv2.copyMakeBorder(ff, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(ff, None, (0, 0), 255)  # Flooding from outside the boundary
        ff = ff[1:-1, 1:-1]
        holes = cv2.bitwise_not(ff) & cv2.bitwise_not(mask)  # # External area
        filled = cv2.bitwise_or(mask, cv2.bitwise_not(holes))

        # ---- Remove small connected regions (to prevent isolated points from affecting the outer contour)----
        num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        min_area_px = (k * k) * 2  # Area threshold: related to the kernel
        clean = np.zeros_like(filled)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                clean[labels == i] = 255

        # findContours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Drawing various contour processing results
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(img_contours, [approx], 0, (0, 255, 255), 2)

        # Display contours and advanced processing results
        plt.figure(figsize=(6, 6))  # Contours
        plt.imshow(img_contours)
        plt.title("Contour Dectection Result: " + plane_name)
        plt.axis("off")
        os.makedirs("Image", exist_ok=True)
        plt.savefig(
            f"Image\\Coutour_{plane_name}.svg", format="svg", bbox_inches="tight"
        )
        if no_image == False and show_plt_contours_Px_2d == True:
            plt.show()
        else:
            plt.close()

        # ---- Convert the contour back to the original coordinate space. ---- #
        contours_real = []  # Each contour in the 2D projection coordinate system
        polygons_2d = []  # shapely Polygon list
        # contour_points_list = []
        linesets = []

        for cnt in contours:
            # Approximate polygon contour
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

            # Converting back from the image coordinate system to the projected two-dimensional coordinate system
            points_2d_back = (
                approx.astype(np.float32) - contour_image_padding
            ) / img_scale + min_pt

            # Mapping from two-dimensional coordinates back to the original three-dimensional space
            points_3d = np.dot(points_2d_back, np.vstack([dir1, dir2])) + center

            contours_real.append(points_2d_back)
            if points_2d_back.size <= 4:
                polygons_2d.append(Polygon())
            else:
                polygons_2d.append(Polygon(points_2d_back))
            # contour_points_list.append(points_3d)

            # Construct LineSet
            num_points = points_3d.shape[0]
            lines = [[i, (i + 1) % num_points] for i in range(num_points)]

            colors = [[1, 0, 0] for _ in lines]  # Red lines

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_3d)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            linesets.append(line_set)

        # Original point cloud color setting for observation
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

        if no_image == False:
            if show_P_contour_3d == True:
                o3d.visualization.draw_geometries(
                    [pcd, *linesets], window_name="Plane Contours 3D View"
                )

        return polygons_2d

    plane_contour_polygon_list = [
        get_plane_contour_polygon(overlap_pcd, dir1, dir2, center, "Plane 1"),
        get_plane_contour_polygon(proj_pcd_p2, dir1, dir2, center, "Plane 2"),
        get_plane_contour_polygon(proj_pcd_p3, dir1, dir2, center, "Plane 3"),
        get_plane_contour_polygon(proj_pcd_p4, dir1, dir2, center, "Plane 4"),
        get_plane_contour_polygon(proj_pcd_p5, dir1, dir2, center, "Plane 5"),
    ]

    # plane_contour_polygon_list = [polygon_p1,polygon_p2,polygon_p3,polygon_p4]

    # ********************** Loop Find feasible TCP *********************
    GRID_SIZE = 1e-9

    def _clean_geom(geom, name=""):
        if geom.is_empty:
            return geom

        g = geom

        # 1) Repair ineffective
        if not g.is_valid:
            g = make_valid(g)
        if not g.is_valid:
            g = g.buffer(0)

        # 2) set_precision：It's safer to put it at the back.
        try:
            g = set_precision(g, GRID_SIZE)
        except Exception as e:
            print(f"[WARN] set_precision failed on '{name}': {e}")

        # 3) merger
        try:
            if hasattr(g, "geoms"):
                g = unary_union(g)
        except Exception:
            pass

        # 4) Final validity check
        if not g.is_valid:
            msg = explain_validity(g)
            print(f"[WARN] Geometry '{name}' still invalid after cleaning: {msg}")

        return g

    def find_feasible_tcp(plane_contour_polygon_list, all_shapes):
        filtered_shapes = []
        feasible_points_on_edge = []
        intersection_areas_on_edge = []
        min_area = 0.15 * (z_pg - 2 * rj) * (b_pg - 2 * rj)

        # First unify the Polygons in plane_contour_polygon_list into lists and clean them
        for i in range(5):
            lst = plane_contour_polygon_list[i]
            if isinstance(lst, Polygon):
                plane_contour_polygon_list[i] = [lst]
            # Clean and set precision for each polygon
            plane_contour_polygon_list[i] = [
                _clean_geom(p, f"plane_poly_{i}") for p in plane_contour_polygon_list[i]
            ]

        poly_p1_list = plane_contour_polygon_list[0]
        poly_p2_list = plane_contour_polygon_list[1]
        poly_p3_list = plane_contour_polygon_list[2]
        poly_p4_list = plane_contour_polygon_list[3]
        poly_p5_list = plane_contour_polygon_list[4]

        for segment_shapes in all_shapes:
            filtered_segment = []
            feasible_point = []
            intersection_areas = []
            for shape in segment_shapes:
                pt = shape["point"]
                rectangles = shape["rectangles"]

                point_geom = Point(pt)
                rect1_geom = Polygon(rectangles[0])  # Finger tip Safe Space
                rect2_geom = Polygon(rectangles[1])  # Finger length
                rect3_geom = Polygon(rectangles[2])  # Gripper Base
                rect4_geom = Polygon(rectangles[3])  # Robot arm
                rect5_geom = Polygon(rectangles[4])  # Gripper Area
                rect6_geom = Polygon(rectangles[5])  # Robot back sapace Box

                total_intersection_areas = sum(
                    poly.intersection(rect5_geom).area for poly in poly_p1_list
                )

                # condition_1 = any(poly.contains(p oint_geom) for poly in poly_0_list)
                condition_1 = total_intersection_areas > min_area
                condition_2 = all(
                    not poly.intersects(rect3_geom) and not poly.intersects(rect4_geom)
                    for poly in poly_p2_list
                )
                condition_3 = all(
                    not poly.intersects(rect1_geom) and not poly.intersects(rect2_geom)
                    for poly in poly_p3_list
                )
                condition_4 = all(
                    not poly.intersects(rect3_geom) and not poly.intersects(rect4_geom)
                    for poly in poly_p4_list
                )
                condition_5 = all(
                    not poly.intersects(rect4_geom) for poly in poly_p5_list
                )

                if (
                    condition_1
                    and condition_2
                    and condition_3
                    and condition_4
                    and condition_5
                ):
                    filtered_segment.append(shape)
                    feasible_point.append(pt)
                    intersection_areas.append(total_intersection_areas)

            filtered_shapes.append(filtered_segment)
            feasible_points_on_edge.append(feasible_point)
            intersection_areas_on_edge.append(intersection_areas)
        return filtered_shapes, feasible_points_on_edge, intersection_areas_on_edge

    feasible_TCP_and_shapes, feasible_TCP, intersection_areas = find_feasible_tcp(
        plane_contour_polygon_list, points_and_gripper_bounding_box
    )

    # *********************** Ranking function ******************************************

    def project_pts_to_3d(points, center, dir1, dir2):
        center = np.asarray(center, dtype=float)
        dir1 = np.asarray(dir1, dtype=float)
        dir2 = np.asarray(dir2, dtype=float)

        basis = np.vstack([dir1, dir2])
        points_list_3d = []

        for pts in points:
            if not pts:
                points_list_3d.append(np.array([], dtype=float))
                continue
            uv = np.asarray(pts, dtype=float).reshape(-1, 2)
            p_3d = center + uv @ basis
            points_list_3d.append(p_3d)

        return points_list_3d

    def get_area_score(intersection_areas):
        area_scores = []
        max_area = max((z_pg - 2 * rj) * (b_pg - 2 * rj), 1e-9)  # avoid 0
        for areas in intersection_areas:
            arr = np.asarray(areas, dtype=float)
            s = (arr - 0.15 * max_area) / (0.85 * max_area)
            s = np.clip(s, 0.0, 1.0)  # a<0.15*max -> 0；a>max -> 1
            area_scores.append(s)
        return area_scores

    def get_center_score(TCP_points, center_pcd):
        center_pcd = np.asarray(center_pcd, dtype=float)
        TCP_points_dist = []

        for pts in TCP_points:
            if pts.size == 0:
                TCP_points_dist.append(np.array([], dtype=float))
                continue
            dist = np.linalg.norm(pts - center_pcd, axis=1)
            TCP_points_dist.append(dist)

        non_empty = [d for d in TCP_points_dist if d.size > 0]
        if len(non_empty) == 0:
            return [d.copy() for d in TCP_points_dist]
        max_dist = np.max([np.max(d) for d in non_empty])

        if max_dist == 0:
            # all points are the same at center of mass -> give all 1.0
            TCP_dist_scores = [np.ones_like(d) for d in TCP_points_dist]
        else:
            TCP_dist_scores = [(1.0 - d / max_dist) for d in TCP_points_dist]

        return TCP_dist_scores

    def rank_feasible_tcp(feasible_TCP, intersection_areas):
        w1, w2 = 0.1, 0.9
        # 2D mean of original_points
        means = np.mean(original_points, axis=0)

        area_scores = get_area_score(intersection_areas)  # list[list[float]]

        tcp_3d = project_pts_to_3d(feasible_TCP, center, dir1, dir2)
        center_scores = get_center_score(tcp_3d, means)  # list[list[float]]

        ranked = []
        for c_seg, a_seg in zip(center_scores, area_scores):
            # ensure same length per segment (should match by construction)
            m = min(len(c_seg), len(a_seg))
            ranked.append([w1 * c_seg[k] + w2 * a_seg[k] for k in range(m)])
        return ranked

    feasible_TCP_rank = rank_feasible_tcp(feasible_TCP, intersection_areas)

    def highlight_feasible_tcp(TCP_points, TCP_rank, segments_2d, tcp_box):
        """
        Iterate through each shape in filtered_shapes and highlight in sequence:
        - All line segments (blue)
        - TCP rectangle of the current point (green)
        - Current point coordinates (red)

        Parameters:
        - TCP_points: List of TCP points to evaluate
        - TCP_rank: Ranking or feasibility scores for TCP points
        - segments_2d: List of 2D line segments [(pt1, pt2), ...]
        - tcp_box: TCP rectangle box for visualization
        """

        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - plt_graphic_padding
        max_xy = all_pts.max(axis=0) + plt_graphic_padding

        for i, pt in enumerate(TCP_points):
            fig, ax = plt.subplots(figsize=(8, 8))  # feasible each edge
            ax.set_title(f"Edge {i + 1}: Feasible TCP and TCP Box")

            # Edges:Blue
            used_labels = set()
            lbl = "Coutours of Plane 2"
            for j, (pt1, pt2) in enumerate(segments_2d):
                if j == i:
                    mid = (pt1 + pt2) / 2
                    vec_12 = pt2 - pt1
                    vec_12 = vec_12 / np.linalg.norm(vec_12)
                    normal_clockwise_90 = [vec_12[1], -vec_12[0]]
                    normal_clockwise_90 = normal_clockwise_90 / np.linalg.norm(
                        normal_clockwise_90
                    )

                    # tilt symbol
                    start_point_line = (
                        mid - normal_clockwise_90 * tilt_symbol_start_dist
                    )
                    end_point_line = (
                        start_point_line
                        + normal_clockwise_90 * tilt_symbol_handle_length
                    )
                    end_point_base1 = (
                        end_point_line
                        + vec_12 * tilt_symbol_finger_width_half
                        - normal_clockwise_90 * tilt_symbol_finger_width_half
                    )
                    end_point_base2 = (
                        end_point_line
                        - vec_12 * tilt_symbol_finger_width_half
                        + normal_clockwise_90 * tilt_symbol_finger_width_half
                    )
                    end_point_finger1 = (
                        end_point_base1
                        + normal_clockwise_90 * tilt_symbol_finger_end_length
                    )
                    end_point_finger2 = (
                        end_point_base2
                        + normal_clockwise_90 * tilt_symbol_finger_end_length
                    )

                    ax.plot(
                        [start_point_line[0], end_point_line[0]],
                        [start_point_line[1], end_point_line[1]],
                        "m",
                        linewidth=1.5,
                        label="Gripper Direction",
                    )
                    ax.plot(
                        [end_point_base1[0], end_point_base2[0]],
                        [end_point_base1[1], end_point_base2[1]],
                        "m",
                        linewidth=1.5,
                    )
                    ax.plot(
                        [end_point_base1[0], end_point_finger1[0]],
                        [end_point_base1[1], end_point_finger1[1]],
                        "m",
                        linewidth=1.5,
                    )
                    ax.plot(
                        [end_point_base2[0], end_point_finger2[0]],
                        [end_point_base2[1], end_point_finger2[1]],
                        "m",
                        linewidth=1.5,
                    )

                if lbl not in used_labels:
                    ax.plot(
                        [pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3, label=lbl
                    )
                    used_labels.add(lbl)
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3)

            # Current TCP: Green
            pt = np.array(pt).reshape(-1, 2)
            if pt.size:
                scores_i = np.array(TCP_rank[i], dtype=float)  # (Ni,)
                if scores_i.shape[0] != pt.shape[0]:
                    print(
                        f"[warn] edge {i}: #scores({scores_i.shape[0]}) != #pts({pt.shape[0]})"
                    )
                    m = min(scores_i.shape[0], pt.shape[0])
                    pt = pt[:m]
                    scores_i = scores_i[:m]
                # ax.plot(pt[:,0], pt[:,1],linestyle='None', marker='x', color='lime', label='Feasible TCP Point')
                scatter = plt.scatter(
                    pt[:, 0],
                    pt[:, 1],
                    c=scores_i,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1,
                    s=5,
                    label="Feasible TCP Point",
                )

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                cbar = plt.colorbar(scatter, cax=cax, label="Score")
                cbar.set_ticks([0.0, 0.5, 1.0])
                cbar.set_ticklabels(["0.0", "0.5", "1.0"])
            else:
                print("No feasible TCP point found!")
                ax.plot(
                    [],
                    [],
                    linestyle="None",
                    marker="x",
                    color="lime",
                    label="Feasible TCP Point",
                )
                # scatter = plt.scatter([], [], c=scores_i, cmap='RdYlGn', s=100, label='Feasible TCP Point')

            # Current TCP Box: Green dotted line
            rect = np.array(tcp_box[i] + [tcp_box[i][0]])
            ax.plot(rect[:, 0], rect[:, 1], "g--", linewidth=0.5, label="TCP Box")

            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            os.makedirs("Image", exist_ok=True)
            plt.savefig(
                f"Image\\feasible_tcp_edge_{i + 1}.svg",
                format="svg",
                bbox_inches="tight",
            )
            plt.show()

    if no_image == False:
        if show_feasible_each_edge == True:
            highlight_feasible_tcp(
                feasible_TCP, feasible_TCP_rank, contour_segments_2d_p2, tcp_box
            )

    def highlight_feasible_all_tcp(TCP_points, TCP_rank, segments_2d, tcp_box):
        """
        Display:
        - All line segments (blue)
        - Feasible TCP points for each edge (colored by score)
        - TCP bounding boxes for each edge (green dashed lines)

        Parameters:
        - TCP_points: List[np.ndarray (Ni,2)], TCP point sets for each edge
        - TCP_rank: List[np.ndarray (Ni,)], scores corresponding to TCP_points
        - segments_2d: List of (pt1, pt2), two endpoints for each edge (2,)
        - tcp_box: List[np.ndarray (4,2)] or (M,2), four rectangle points for each edge (in order)
        """

        # Collect all points to set the coordinate range
        all_xy = []
        for pt1, pt2 in segments_2d:
            all_xy.extend([pt1, pt2])
        for pts in TCP_points:
            if pts is not None and len(pts) > 0:
                all_xy.extend(list(np.asarray(pts)))
        for rect in tcp_box:
            r = np.asarray(rect)
            if r.ndim == 2 and r.shape[0] >= 3:
                all_xy.extend(list(r))

        all_xy = np.asarray(all_xy) if len(all_xy) else np.zeros((1, 2))
        min_xy = all_xy.min(axis=0) - plt_graphic_padding
        max_xy = all_xy.max(axis=0) + plt_graphic_padding

        fig, ax = plt.subplots(figsize=(8, 6))  # feasible all tcp
        ax.set_title("All Feasible TCP and TCP Box")

        used_labels = set()
        scatter_handle = None
        cbar = None

        # Synchronously traverse all four to avoid index misalignment.
        for (pt1, pt2), tcp, rect, scores in zip(
            segments_2d, TCP_points, tcp_box, TCP_rank
        ):
            # 1) Edge Blue
            lbl = "Contours on Plane"
            if lbl not in used_labels:
                ax.plot(
                    [pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3, label=lbl
                )
                used_labels.add(lbl)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "b-", linewidth=3)

            # 2) TCP Point -Color by score
            tcp = np.asarray(tcp) if tcp is not None else np.empty((0, 2))
            scores = (
                np.asarray(scores, dtype=float)
                if scores is not None
                else np.empty((0,))
            )
            if tcp.ndim == 1 and tcp.size == 2:
                tcp = tcp.reshape(1, 2)

            if tcp.size > 0:
                # alignment length
                m = min(len(tcp), len(scores))
                if m == 0:
                    pass
                else:
                    tcp = tcp[:m]
                    scores = scores[:m]
                    vmin = 0.0
                    vmax = 1.0
                    if vmin == vmax:
                        vmin, vmax = vmax - 1.0, vmax + 1.0

                    lbl = "Feasible TCP Point"
                    scatter_handle = ax.scatter(
                        tcp[:, 0],
                        tcp[:, 1],
                        c=scores,
                        cmap="RdYlGn",
                        marker="x",
                        vmin=vmin,
                        vmax=vmax,
                        s=20,
                        label=(
                            lbl if "Feasible TCP Point" not in used_labels else None
                        ),
                    )
                    if "Feasible TCP Point" not in used_labels:
                        used_labels.add("Feasible TCP Point")

                    if cbar is None and scatter_handle is not None:
                        # cbar = plt.colorbar(scatter_handle, ax=ax, label='Score', fraction=0.046, pad=0.04)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.3)
                        cbar = plt.colorbar(scatter_handle, cax=cax, label="Score")
                        cbar.set_ticks([0.0, 0.5, 1.0])
                        cbar.set_ticklabels(["0.0", "0.5", "1.0"])

            # 3) TCP Box Green dotted line
            rect = np.asarray(rect)
            if rect.ndim == 2 and rect.shape[0] >= 3:
                rect_closed = np.vstack([rect, rect[0]])
                lbl = "TCP Box"
                if lbl not in used_labels:
                    ax.plot(
                        rect_closed[:, 0],
                        rect_closed[:, 1],
                        "g--",
                        linewidth=0.5,
                        label=lbl,
                    )
                    used_labels.add(lbl)
                else:
                    ax.plot(rect_closed[:, 0], rect_closed[:, 1], "g--", linewidth=0.5)

        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        plt.tight_layout()
        os.makedirs("Image", exist_ok=True)
        plt.savefig("Image\\feasible_tcp_all_2d.svg", format="svg", bbox_inches="tight")
        plt.show()

    if no_image == False:
        if show_all_feasbile_in_2d == True:
            highlight_feasible_all_tcp(
                feasible_TCP, feasible_TCP_rank, contour_segments_2d_p2, tcp_box
            )

    def show_feasible_tcp_in_3d(
        TCP_3d, TCP_rank, segments_3d_para, pcd_pa, pcd_pb, pcd
    ):
        line_segments_3d, line_indices = segments_3d_para

        if len(line_indices) == 0:
            print(
                "No visible line segments (possibly due to the outline being too small or overly simplified)"
            )
            return

        blue = np.array([[0, 0, 1] for _ in range(len(TCP_3d))], dtype=float)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(
            np.asarray(line_segments_3d, dtype=float)
        )
        line_set.lines = o3d.utility.Vector2iVector(
            np.asarray(line_indices, dtype=np.int32)
        )
        line_set.colors = o3d.utility.Vector3dVector(blue)

        TCP_3d_flat = [p for group in TCP_3d for p in group]
        TCP_rank_flat = [r for group in TCP_rank for r in group]

        colors = plt.cm.RdYlGn(np.asarray(TCP_rank_flat, dtype=float))[:, :3]

        pcd_tcp = o3d.geometry.PointCloud()
        pcd_tcp.points = o3d.utility.Vector3dVector(TCP_3d_flat)
        pcd_tcp.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries(
            [pcd_tcp, pcd_pb, line_set], window_name="Feasible TCP in 3D with PA and PB"
        )

        o3d.visualization.draw_geometries(
            [pcd_tcp, pcd, line_set],
            window_name="Feasible TCP in 3D wit assembled part",
        )

    feasible_TCP_3d = project_pts_to_3d(feasible_TCP, center, dir1, dir2)

    if no_image == False:
        if show_feasible_with_P_and_pcd == True:
            show_feasible_tcp_in_3d(
                feasible_TCP_3d,
                feasible_TCP_rank,
                contour_segments_3d_p2_para,
                pcd_orig_i,
                pcd_orig_j,
                actual_pcd,
            )
