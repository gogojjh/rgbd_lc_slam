from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


def build_submap(keyframe_pcds_w: list[o3d.geometry.PointCloud], *, submap_k: int, voxel: float) -> o3d.geometry.PointCloud:
    """Build a local submap from last K keyframes.

    Inputs are expected in *world frame*.
    """

    if len(keyframe_pcds_w) == 0:
        return o3d.geometry.PointCloud()

    sub = o3d.geometry.PointCloud()
    for pcd in keyframe_pcds_w[-submap_k:]:
        sub += pcd

    if len(sub.points) == 0:
        return sub

    sub = sub.voxel_down_sample(voxel)
    sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    return sub


@dataclass
class SubmapState:
    keyframe_pcds_w: list[o3d.geometry.PointCloud]
    keyframe_Twc: list[np.ndarray]
