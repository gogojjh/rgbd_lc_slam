from __future__ import annotations

import numpy as np
import open3d as o3d


def build_submap(
    keyframe_pcds: list[o3d.geometry.PointCloud],
    *,
    submap_k: int,
    voxel: float,
) -> o3d.geometry.PointCloud:
    if not keyframe_pcds:
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    for p in keyframe_pcds[-submap_k:]:
        pcd += p

    if len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30)
        )

    return pcd


def should_add_keyframe(
    T_prev: np.ndarray,
    T_cur: np.ndarray,
    *,
    keyframe_trans: float,
    keyframe_rot_deg: float,
) -> bool:
    dT = np.linalg.inv(T_prev) @ T_cur
    trans = float(np.linalg.norm(dT[:3, 3]))

    # compute rotation angle
    R = dT[:3, :3]
    angle = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    angle_deg = float(angle * 180.0 / np.pi)

    return (trans > keyframe_trans) or (angle_deg > keyframe_rot_deg)
