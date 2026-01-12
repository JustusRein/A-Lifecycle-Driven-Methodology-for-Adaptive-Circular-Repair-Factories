from typing import Literal, TypeAlias, TypedDict, Union
import numpy as np

import open3d as o3d
import trimesh

# Type Aliases
PointCloud: TypeAlias = o3d.geometry.PointCloud
TriangleMesh: TypeAlias = o3d.geometry.TriangleMesh
Tensor: TypeAlias = o3d.core.Tensor
# Type aliases for clarity
Geometry3D = Union[
    o3d.geometry.TriangleMesh,
    o3d.geometry.PointCloud,
    trimesh.Trimesh,
    trimesh.PointCloud,
]

MeshTypes = Union[
    trimesh.Trimesh,
    o3d.geometry.TriangleMesh,
]

PointCloudTypes = Union[
    o3d.geometry.PointCloud,
    trimesh.PointCloud,
]

TrimeshTypes = Union[
    trimesh.Trimesh,
    trimesh.PointCloud,
]

Open3DTypes = Union[
    o3d.geometry.TriangleMesh,
    o3d.geometry.PointCloud,
]

PathLike = str
Backend = Literal["open3d", "trimesh"]


class DimensionsReport(TypedDict):
    extents_xyz: np.ndarray
    diagonal: float
    likely_unit: Literal["meters", "millimeters"]
    suggested_voxel_size: float
    details: str
