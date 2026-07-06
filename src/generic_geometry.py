import open3d as o3d
import trimesh
import numpy as np
import os
from typing import Optional, Literal, Callable
from functools import wraps
from src.utils import geometry_utils as gu
from src.utils.types import (
    DimensionsReport,
    Geometry3D,
    Open3DTypes,
    TrimeshTypes,
    PathLike,
    Backend,
)


def inplace_result(func: Callable):
    """
    Decorator to handle 'inplace' operations.
    If 'inplace=True' is passed to the decorated method, the internal state
    (self.geometry) is automatically updated with the result.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        inplace = kwargs.pop("inplace", False)

        result_geometry = func(self, *args, **kwargs)

        if inplace:
            if result_geometry is not None:
                self.set_geometry(result_geometry)
            else:
                pass
                # print(
                #     f"Warning: Method {func.__name__} returned None, inplace update ignored."
                # )

        return result_geometry

    return wrapper


class GenericGeometry:
    """
    A utility class to handle loading, converting, processing, and saving 3D geometry data.
    Supports Open3D and Trimesh backends with robust type conversion.
    """

    def __init__(
        self, filepath: Optional[PathLike] = None, geometry: Optional[Geometry3D] = None
    ):
        if filepath is not None:
            self.load_file(filepath)
        elif geometry is not None:
            self.set_geometry(geometry)
        else:
            self.geometry: Optional[Geometry3D] = None
            self.geometry_type: Optional[str] = None  # 'mesh' or 'point_cloud'
            self.backend: Optional[Backend] = None

    def set_geometry(self, geometry: Geometry3D) -> None:
        """
        Manually sets the internal geometry object.
        Automatically infers the backend and geometry type based on the object instance.
        """
        # Infer Backend and Type using isinstance checks against library classes
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            self.geometry_type = "mesh"
            self.backend = "open3d"
        elif isinstance(geometry, o3d.geometry.PointCloud):
            self.geometry_type = "point_cloud"
            self.backend = "open3d"
        elif isinstance(geometry, trimesh.Trimesh):
            self.geometry_type = "mesh"
            self.backend = "trimesh"
        elif isinstance(geometry, trimesh.PointCloud):
            self.geometry_type = "point_cloud"
            self.backend = "trimesh"
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry).__name__}")

        self.geometry = geometry
        # print(f"✅ Geometry set: {self.geometry_type} ({self.backend})")

    def load_file(
        self,
        file_path: PathLike,
        backend: Backend = "open3d",
        expected_type: Literal["auto", "mesh", "point_cloud"] = "auto",
    ) -> Geometry3D:
        """
        Loads a 3D file using the specified backend without modifying the data.
        Updates the internal state of the class.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loaded_geom = None
        if backend == "open3d":
            loaded_geom = self._load_open3d(file_path, expected_type)
        elif backend == "trimesh":
            loaded_geom = self._load_trimesh(file_path, expected_type)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.set_geometry(loaded_geom)
        return loaded_geom

    # --- Processing Methods (Decorated for Inplace Support) ---

    @inplace_result
    def to_point_cloud(self, num_points: int = 5000) -> o3d.geometry.PointCloud:
        """
        Converts the current geometry to an Open3D PointCloud.
        If the current geometry is a Mesh, it samples points using Poisson Disk Sampling.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        geom_o3d = self._as_open3d()

        if self.geometry_type == "point_cloud":
            return geom_o3d  # Already a PCD

        # print(f"Sampling {num_points} points from mesh...")
        pcd = geom_o3d.sample_points_poisson_disk(num_points)
        return pcd

    @inplace_result
    def to_mesh_convex_hull(self) -> o3d.geometry.TriangleMesh:
        """
        Converts the current geometry to a Convex Hull Mesh (Open3D).
        Useful for fast collision checking or bounding volumes.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        geom_o3d = self._as_open3d()

        # print("Computing Convex Hull...")
        hull, _ = geom_o3d.compute_convex_hull()
        hull.compute_vertex_normals()
        return hull

    @inplace_result
    def to_mesh_poisson(
        self, depth: int = 9, cleanup: bool = True
    ) -> o3d.geometry.TriangleMesh:
        """
        Reconstructs a mesh using Poisson Surface Reconstruction.
        Requires normals (computes them if missing).
        """
        geom_o3d = self._as_open3d()

        # Poisson requires a Point Cloud. If mesh, sample it first.
        if self.geometry_type == "mesh":
            pcd = geom_o3d.sample_points_poisson_disk(5000)
        else:
            pcd = geom_o3d

        if not pcd.has_normals():
            # print("Estimating normals for Poisson...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01, max_nn=30
                )
            )
        pts = np.asarray(pcd.points)
        cx, cy = np.mean(pts, axis=0)[:2]
        max_z = np.max(pts[:, 2])
        camera_z = max_z * 2.0 if max_z > 0 else max_z + 0.5
        camera_pos = np.array([cx, cy, camera_z])
        pcd.orient_normals_towards_camera_location(camera_pos)
        # pcd.orient_normals_consistent_tangent_plane(15)

        # print(f"Running Poisson reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

        if cleanup:
            # print("Cleaning up low density artifacts...")
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh

    @inplace_result
    def downsample(self, voxel_size: float) -> o3d.geometry.PointCloud:
        """
        Reduces the resolution of the geometry using a Voxel Grid filter.
        If input is a Mesh, it samples it densely first to preserve shape before downsampling.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        geom_o3d = self._as_open3d()

        # If Mesh, sample densely first to capture shape, then downsample
        if self.geometry_type == "mesh":
            # print("Input is Mesh. Sampling surface before downsampling...")
            pcd = geom_o3d.sample_points_poisson_disk(number_of_points=100000)
        else:
            pcd = geom_o3d

        # print(f"Downsampling with voxel_size={voxel_size}...")
        pcd_down = pcd.voxel_down_sample(voxel_size)
        return pcd_down

    # --- Conversion to Trimesh ---

    def as_trimesh(self) -> TrimeshTypes:
        """
        Returns the current geometry as a Trimesh object.
        Does NOT modify internal state.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded.")

        if self.backend == "trimesh":
            return self.geometry

        # print("🔄 Converting Open3D object to Trimesh...")

        if self.geometry_type == "mesh":
            return self._open3d_mesh_to_trimesh(self.geometry)
        elif self.geometry_type == "point_cloud":
            return self._open3d_pcd_to_trimesh(self.geometry)
        else:
            raise TypeError(f"Unknown geometry type: {self.geometry_type}")

    # --- IO & Helpers ---

    def save(self, output_path: PathLike, write_ascii: bool = False) -> None:
        """
        Saves the current geometry to disk. Format is inferred from extension.
        """
        if self.geometry is None:
            raise ValueError("No geometry to save")

        # Ensure we save using Open3D for consistency
        geom_o3d = self._as_open3d()
        # print(f"💾 Saving to: {output_path}")

        if self.geometry_type == "point_cloud":
            o3d.io.write_point_cloud(output_path, geom_o3d, write_ascii=write_ascii)
        else:
            geom_o3d.compute_vertex_normals()
            o3d.io.write_triangle_mesh(output_path, geom_o3d, write_ascii=write_ascii)

    def visualize(
        self, window_name: str = "Geometry Viewer", show_frame: bool = True
    ) -> None:
        """
        Opens a visualization window using Open3D.
        """
        if self.geometry is None:
            # print("❌ Nothing to visualize.")
            return

        geom = self._as_open3d()
        geoms_to_draw = [geom]

        if show_frame:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )
            geoms_to_draw.append(frame)

        o3d.visualization.draw_geometries(geoms_to_draw, window_name=window_name)

    # --- Internal Loaders & Converters ---

    def _load_open3d(self, path: PathLike, type_hint: str) -> Open3DTypes:
        """Internal loader for Open3D backend."""
        if type_hint in ["auto", "mesh"]:
            try:
                mesh = o3d.io.read_triangle_mesh(path)
                if not mesh.is_empty() and len(mesh.triangles) > 0:
                    # print(f"[Open3D] Loaded Mesh: {path}")
                    return mesh
            except Exception:
                pass

        if type_hint in ["auto", "point_cloud"]:
            try:
                pcd = o3d.io.read_point_cloud(path)
                if not pcd.is_empty():
                    # print(f"[Open3D] Loaded PointCloud: {path}")
                    return pcd
            except Exception:
                pass

        raise ValueError(f"Failed to load {path} with Open3D as {type_hint}")

    def _load_trimesh(self, path: PathLike, type_hint: str) -> TrimeshTypes:
        """Internal loader for Trimesh backend."""
        try:
            geom = trimesh.load(path)

            is_mesh = isinstance(geom, trimesh.Trimesh)
            is_pcd = isinstance(geom, trimesh.PointCloud)

            if type_hint == "mesh" and not is_mesh:
                raise ValueError("Object is not a mesh")
            if type_hint == "point_cloud" and not is_pcd:
                raise ValueError("Object is not a point cloud")

            if is_mesh:
                # print(f"[Trimesh] Loaded Mesh: {path}")
                pass
            else:
                # print(f"[Trimesh] Loaded PointCloud: {path}")
                pass

            assert isinstance(geom, TrimeshTypes)
            return geom
        except Exception as e:
            raise ValueError(f"Trimesh load failed: {e}")

    def _as_open3d(self) -> Open3DTypes:
        """
        Helper to ensure internal geometry is in Open3D format.
        Handles manual conversion from Trimesh if needed.
        """
        if self.backend == "open3d":
            return self.geometry

        if self.geometry_type == "mesh":
            return self._trimesh_mesh_to_open3d(self.geometry)
        elif self.geometry_type == "point_cloud":
            return self._trimesh_pcd_to_open3d(self.geometry)
        else:
            raise TypeError(
                f"Cannot convert unknown geometry type '{self.geometry_type}' to Open3D."
            )

    def _trimesh_mesh_to_open3d(
        self, geom: trimesh.Trimesh
    ) -> o3d.geometry.TriangleMesh:
        """Manual conversion: Trimesh Mesh -> Open3D TriangleMesh"""
        mesh_o3d = o3d.geometry.TriangleMesh()

        # Use ascontiguousarray to ensure memory is writeable and compatible with C++
        vertices = np.ascontiguousarray(geom.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(geom.faces, dtype=np.int32)

        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

        if "vertex_normals" in geom._cache:
            normals = np.ascontiguousarray(geom.vertex_normals, dtype=np.float64)
            mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(normals)
        else:
            mesh_o3d.compute_vertex_normals()
        return mesh_o3d

    def _trimesh_pcd_to_open3d(
        self, geom: trimesh.PointCloud
    ) -> o3d.geometry.PointCloud:
        """Manual conversion: Trimesh PointCloud -> Open3D PointCloud"""
        pcd_o3d = o3d.geometry.PointCloud()

        vertices = np.ascontiguousarray(geom.vertices, dtype=np.float64)
        pcd_o3d.points = o3d.utility.Vector3dVector(vertices)

        if hasattr(geom, "colors") and geom.colors is not None:
            # Take only RGB channels, ignore Alpha
            colors = np.ascontiguousarray(geom.colors[:, :3], dtype=np.float64) / 255.0
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

        return pcd_o3d

    def _open3d_mesh_to_trimesh(
        self, geom: o3d.geometry.TriangleMesh
    ) -> trimesh.Trimesh:
        """Manual conversion: Open3D TriangleMesh -> Trimesh"""
        vertices = np.asarray(geom.vertices)
        faces = np.asarray(geom.triangles)

        # process=False prevents automatic merging/altering of data
        mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        if geom.has_vertex_normals():
            mesh_tri.vertex_normals = np.asarray(geom.vertex_normals)

        return mesh_tri

    def _open3d_pcd_to_trimesh(
        self, geom: o3d.geometry.PointCloud
    ) -> trimesh.PointCloud:
        """Manual conversion: Open3D PointCloud -> Trimesh"""
        points = np.asarray(geom.points)
        colors = None

        if geom.has_colors():
            # Convert back to 0-255 uint8
            c = np.asarray(geom.colors) * 255.0
            colors = c.astype(np.uint8)
            # Trimesh prefers RGBA
            if colors.shape[1] == 3:
                alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
                colors = np.hstack((colors, alpha))

        pcd_tri = trimesh.PointCloud(vertices=points, colors=colors)
        return pcd_tri

    # ==============================================================================
    # GEOMETRIC UTILS METHODS (Delegating to geometry_utils)
    # ==============================================================================

    @inplace_result
    def transform(
        self, translation: np.ndarray, rotation_axis_angle: Optional[np.ndarray] = None
    ) -> Geometry3D:
        """
        Applies a rigid transformation to the geometry.
        Uses geometry_utils to create the matrix if rotation is provided.

        Args:
            translation: [x, y, z] vector in meters.
            rotation_axis_angle: Optional [angle_deg, axis_x, axis_y, axis_z].
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        geom_o3d = self._as_open3d()

        # 1. Create Transformation Matrix (T)
        if rotation_axis_angle is not None:
            # Use utility from geometry_utils
            T = gu.create_transformation_matrix(rotation_axis_angle, translation)
        else:
            # Simple translation
            T = np.eye(4)
            T[:3, 3] = translation

        # 2. Apply Transformation
        # print("Applying rigid transformation...")
        return geom_o3d.transform(T)

    @inplace_result
    def apply_transform(self, T: np.ndarray) -> Geometry3D:
        """
        Applies a 4x4 homogeneous transformation matrix to the geometry.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        geom_o3d = self._as_open3d()
        return geom_o3d.transform(T)

    def align_to(
        self,
        reference_geometry,
        voxel_size: Optional[float] = None,
        num_samples: int = 10000,
        inplace: bool = False,
    ) -> np.ndarray:
        """
        Aligns this geometry (source) to the reference_geometry (target).
        Returns the 4x4 transformation matrix that, applied to this geometry,
        puts it in the same reference frame as the reference_geometry.

        Args:
            reference_geometry: The reference geometry (target).
            voxel_size: Voxel size for downsampling.
            num_samples: Number of points to sample if geometry is a mesh.
            inplace: If True, applies the estimated transformation to this geometry.
        """
        from src.utils.geometry_utils import align_point_clouds

        T = align_point_clouds(
            target_geom=reference_geometry,
            source_geom=self,
            voxel_size=voxel_size,
            num_samples=num_samples,
        )

        if inplace:
            self.apply_transform(T, inplace=True)

        return T

    @inplace_result
    def filter_by_normal(
        self, direction: np.ndarray, angle_tolerance: float = 15.0
    ) -> o3d.geometry.PointCloud:
        """
        Filters points whose normals align with a specific direction.
        Delegates to geometry_utils.filter_pcd_by_normal_direction.

        Args:
            direction: [x, y, z] vector (e.g., [0, 0, 1] for Z+).
            angle_tolerance: Max angle difference in degrees.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        # This operation requires a Point Cloud with normals
        pcd = self.to_point_cloud(inplace=False)

        # Calculate cosine threshold from degrees
        cos_threshold = np.cos(np.radians(angle_tolerance))

        # print(f"Filtering by normal direction {direction} (tol={angle_tolerance}°)...")

        # Delegate to utility function
        filtered_pcd, _ = gu.filter_pcd_by_normal_direction(
            pcd, n_ref=direction, cos_threshold=cos_threshold
        )

        return filtered_pcd

    @inplace_result
    def remove_outliers(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ) -> o3d.geometry.PointCloud:
        """
        Removes noisy points using Statistical Outlier Removal.
        Delegates to geometry_utils.remove_outliers_statistical.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        pcd = self.to_point_cloud(inplace=False)

        # print(f"Removing statistical outliers (k={nb_neighbors}, std={std_ratio})...")

        # Delegate to utility function
        clean_pcd, _ = gu.remove_outliers_statistical(
            pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

        return clean_pcd

    @inplace_result
    def remove_outliers_dynamic_sor(
        self, std_ratio: float = 3.0, inplace: bool = False
    ):
        """
        Dynamically calculates the number of neighbors based on the cloud size
        to keep the statistical outlier removal scale-invariant.
        """
        num_points = len(self.geometry.points)

        # Calculate 1% of the total points
        dynamic_neighbors = int(num_points * 0.01)

        # Clamp the value between a safe minimum (20) and a safe maximum (100)
        # to prevent performance drops or excessive corrosion on small/large clouds
        nb_neighbors = max(20, min(100, dynamic_neighbors))

        cl, ind = self.geometry.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

        cleaned_pcd = self.geometry.select_by_index(ind)

        if inplace:
            self.geometry = cleaned_pcd

        return cleaned_pcd

    @inplace_result
    def remove_outliers_physical(self, min_neighbors: int = 5, inplace: bool = False):
        """
        Uses Radius Outlier Removal based on the actual physical spacing of the point cloud.
        Excellent for preserving sharp edges like CubeSats or mechanical parts.
        """
        # 1. Discover the "tile size" of the current cloud
        distances = self.geometry.compute_nearest_neighbor_distance()
        avg_spacing = np.mean(np.asarray(distances))

        # 2. Define the search radius as exactly 3 times the average spacing
        # If a point doesn't have neighbors within a 3-tile radius, it's floating dust.
        dynamic_radius = avg_spacing * 3.0

        cl, ind = self.geometry.remove_radius_outlier(
            nb_points=min_neighbors, radius=dynamic_radius
        )

        cleaned_pcd = self.geometry.select_by_index(ind)

        if inplace:
            self.geometry = cleaned_pcd

        return cleaned_pcd

    @inplace_result
    def cluster_dbscan(
        self, eps: float = 0.02, min_points: int = 10, num_points: int = 5000
    ) -> o3d.geometry.PointCloud:
        """
        Clusters the cloud and keeps only the largest cluster (removes floating noise).
        Delegates to geometry_utils.remove_outliers_dbscan.
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        pcd = self.to_point_cloud(inplace=False, num_points=num_points)

        # print(f"Clustering with DBSCAN (eps={eps}, min_pts={min_points})...")

        # Delegate to utility function
        main_cluster = gu.remove_outliers_dbscan(pcd, eps=eps, min_points=min_points)

        return main_cluster

    @inplace_result
    def estimate_normals(
        self,
        radius: Optional[float] = None,
        max_nn: int = 30,
        align_direction: Optional[np.ndarray] = None,
    ) -> o3d.geometry.PointCloud:
        """
        Estimates and orients normals for the geometry.
        Delegates to geometry_utils.compute_pcd_normals.

        Args:
            radius: Search radius. If None, auto-calculated based on object size.
            max_nn: Max neighbors for KDTree search.
            align_direction: Vector [x, y, z] to align normals (e.g., [0, 0, 1] for Z+).
        """
        if self.geometry is None:
            raise ValueError("No geometry loaded")

        # Ensure it's a Point Cloud (Meshes usually have vertex normals, but this recalculates/refines them)
        pcd = self.to_point_cloud(inplace=False)

        # print(f"Estimating normals (radius={radius}, k={max_nn})...")

        # Delegate to utility function
        pcd_normals = gu.compute_pcd_normals(
            pcd, radius=radius, max_nn=max_nn, align_vector=align_direction
        )

        return pcd_normals

    def get_dimensions_report(self) -> DimensionsReport:
        return gu.analyze_object_dimensions(self.geometry)

    @inplace_result
    def clean_point_cloud(
        self,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        voxel_size: float = 0.0,
        normal_orientation: Optional[np.ndarray] = None,
    ):
        """
        This represents the complete 'Preprocessing' pipeline for grasp sampling.
        Applies voxel downsampling (if voxel_size > 0.0), statistical cleaning, and ensures normal consistency.

        Args:
            voxel_size (float): Size of the voxel grid for downsampling.
                                Set to 0.0 to skip downsampling.
            nb_neighbors (int): Number of neighbors to analyze for noise stats.
            std_ratio (float): Threshold. Lower is more aggressive.
        """
        # 0. Ensure it is a Point Cloud
        self.to_point_cloud(inplace=True)
        # print(f"[Clean] Points before processing: {len(self.geometry.points)}")

        # 1. Voxel Downsampling (CRITICAL: Must happen BEFORE statistical removal)
        # Standardizes point density and drastically speeds up the next algorithms
        if voxel_size > 0.0:
            self.geometry = self.geometry.voxel_down_sample(voxel_size=voxel_size)
            # print(f"[Clean] Points after Downsample: {len(self.geometry.points)}")

        # 2. Statistical Outlier Removal
        # Removes "flying pixels" and sparse noise (dust) around the object.
        # It does NOT remove valid geometry like the back of the can.
        pcd_clean, ind = self.geometry.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

        # 3. Re-estimate Normals
        # Necessary because removing points invalidates the old KDTree/Normals.
        # Uses a Hybrid search (Radius + KNN) for robustness.
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

        # 4. Consistent Normal Orientation
        # if orientation is given, point to camera, else do consistent tangent plane
        if normal_orientation is not None:
            pcd_clean.orient_normals_towards_camera_location(normal_orientation)
        else:
            pcd_clean.orient_normals_consistent_tangent_plane(k=15)

        # print(f"[Clean] Points after full cleaning: {len(self.geometry.points)}")
        return pcd_clean

    def scale(self, scale_factor: float, center: bool = False):
        """
        Scales the geometry proportionally by the provided factor.

        Args:
            scale_factor (float): The multiplication factor (e.g., 0.001 to convert mm to m).
            center (bool): If True, scales around the object's Center of Mass/centroid.
                           If False (default), scales around the origin (0,0,0) to keep the TCP fixed.
        """
        import numpy as np
        import open3d as o3d
        import trimesh

        # 1. Determine the central scaling point (center_pt)
        if center:
            if isinstance(self.geometry, o3d.geometry.Geometry3D):
                center_pt = self.geometry.get_center()
            elif isinstance(self.geometry, trimesh.Trimesh):
                center_pt = self.geometry.centroid
            else:
                center_pt = np.array([0.0, 0.0, 0.0])
        else:
            center_pt = np.array([0.0, 0.0, 0.0])

        # 2. Apply scaling according to the geometric engine
        if isinstance(self.geometry, o3d.geometry.Geometry3D):
            # Open3D has a native method that accepts the center point
            self.geometry.scale(scale_factor, center=center_pt)

        elif isinstance(self.geometry, trimesh.Trimesh):
            # For Trimesh, we construct a 4x4 transformation matrix
            # Mathematical equation: p' = S*(p - C) + C  =>  p' = S*p + (C - S*C)
            transform = np.eye(4)
            transform[:3, :3] *= scale_factor
            transform[:3, 3] = center_pt - (scale_factor * center_pt)

            self.geometry.apply_transform(transform)

        else:
            raise TypeError(
                f"Geometry type not supported for scaling: {type(self.geometry)}"
            )

        return self  # Allows method chaining (e.g., geom.scale(0.001).visualize())
