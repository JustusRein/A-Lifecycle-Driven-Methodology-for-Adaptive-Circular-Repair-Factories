import numpy as np
import open3d as o3d
import trimesh
import copy
from src.utils.geometry_utils import align_point_clouds
from src.generic_geometry import GenericGeometry

def generate_cube_point_cloud(num_points=2000) -> o3d.geometry.PointCloud:
    """Generates a point cloud of a cube with some surface coordinates."""
    points = np.random.uniform(-1.0, 1.0, (num_points, 3))
    # Project to cube surface
    for i in range(num_points):
        dim_to_clamp = np.random.randint(0, 3)
        points[i, dim_to_clamp] = np.random.choice([-1.0, 1.0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Estimate initial normals
    pcd.estimate_normals()
    return pcd

def test_align_point_clouds_basic():
    # 1. Generate target point cloud (first reference)
    pcd_target = generate_cube_point_cloud(3000)
    
    # 2. Create a known transformation matrix (Rotation + Translation)
    # Rotation of 45 degrees around Z axis, translation by [0.5, -0.2, 0.1]
    angle = np.radians(45.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    t = np.array([0.5, -0.2, 0.1])
    T_applied = np.eye(4)
    T_applied[:3, :3] = R
    T_applied[:3, 3] = t
    
    # 3. Create source point cloud (second reference) by transforming target
    pcd_source = copy.deepcopy(pcd_target)
    pcd_source.transform(T_applied)
    
    # 4. Align source to target (find T that maps source -> target)
    # Since we mapped target -> source with T_applied, the mapping source -> target
    # should be T_applied^-1
    T_expected = np.linalg.inv(T_applied)
    
    print("[Test] Running alignment on raw Open3D point clouds...")
    T_estimated = align_point_clouds(pcd_target, pcd_source, voxel_size=0.1)
    
    print(f"Expected T:\n{T_expected}")
    print(f"Estimated T:\n{T_estimated}")
    
    # Check that estimated T aligns source to target
    pcd_source_aligned = copy.deepcopy(pcd_source).transform(T_estimated)
    
    # Calculate distances between aligned source and target
    distances = pcd_target.compute_point_cloud_distance(pcd_source_aligned)
    mean_dist = np.mean(distances)
    print(f"Mean alignment error (distance): {mean_dist:.6f}")
    
    assert mean_dist < 0.05, f"Alignment failed, mean distance: {mean_dist}"
    assert np.allclose(T_estimated, T_expected, atol=0.08), "Estimated transformation is too far from expected"
    print("✅ Raw Open3D PointCloud alignment test passed!")

def test_align_point_clouds_generic_geometry():
    # Test using GenericGeometry wrappers
    pcd_target = generate_cube_point_cloud(3000)
    
    # Rotation and translation
    angle = np.radians(30.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    t = np.array([-0.2, 0.3, -0.4])
    T_applied = np.eye(4)
    T_applied[:3, :3] = R
    T_applied[:3, 3] = t
    
    pcd_source = copy.deepcopy(pcd_target)
    pcd_source.transform(T_applied)
    
    # Wrap in GenericGeometry
    geom_target = GenericGeometry(geometry=pcd_target)
    geom_source = GenericGeometry(geometry=pcd_source)
    
    print("[Test] Running alignment on GenericGeometry objects...")
    T_estimated = align_point_clouds(geom_target, geom_source, voxel_size=0.1)
    
    pcd_source_aligned = copy.deepcopy(pcd_source).transform(T_estimated)
    distances = pcd_target.compute_point_cloud_distance(pcd_source_aligned)
    mean_dist = np.mean(distances)
    print(f"GenericGeometry Mean alignment error (distance): {mean_dist:.6f}")
    
    assert mean_dist < 0.05, f"GenericGeometry alignment failed, mean distance: {mean_dist}"
    print("✅ GenericGeometry wrapper alignment test passed!")

    # Test GenericGeometry align_to method with inplace=True
    print("[Test] Running align_to in-place on GenericGeometry...")
    geom_source_copy = GenericGeometry(geometry=copy.deepcopy(pcd_source))
    T_estimated_2 = geom_source_copy.align_to(geom_target, voxel_size=0.1, inplace=True)
    
    # Check that it aligns correctly in-place
    distances_inplace = pcd_target.compute_point_cloud_distance(geom_source_copy.geometry)
    mean_dist_inplace = np.mean(distances_inplace)
    print(f"GenericGeometry in-place alignment error: {mean_dist_inplace:.6f}")
    assert mean_dist_inplace < 0.05, f"GenericGeometry in-place alignment failed, mean distance: {mean_dist_inplace}"
    assert np.allclose(T_estimated, T_estimated_2), "align_to returned different matrix than align_point_clouds"
    print("✅ GenericGeometry.align_to in-place alignment test passed!")

if __name__ == "__main__":
    test_align_point_clouds_basic()
    test_align_point_clouds_generic_geometry()

