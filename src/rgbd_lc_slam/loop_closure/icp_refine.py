from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import CameraIntrinsics


@dataclass
class ICPResult:
    T_ij: np.ndarray
    fitness: float
    rmse_m: float


class ICPRefiner:
    """Optional ICP refinement using Open3D.

    Refines T_ij between two RGB-D frames.

    Note: This requires `open3d`.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        max_corr_dist_m: float = 0.07,
        max_iters: int = 30,
    ):
        self.intr = intrinsics
        self.max_corr = float(max_corr_dist_m)
        self.max_iters = int(max_iters)

    def refine(
        self,
        rgb_i: np.ndarray,
        depth_i_m: np.ndarray,
        rgb_j: np.ndarray,
        depth_j_m: np.ndarray,
        T_ij_init: np.ndarray,
    ) -> Optional[ICPResult]:
        try:
            import open3d as o3d
        except Exception:
            return None

        # Open3D expects uint8 color and depth either in uint16 (mm) or float (m) depending
        # on how we construct RGBD. We use float depth in meters.
        color_i = o3d.geometry.Image(rgb_i.astype(np.uint8))
        color_j = o3d.geometry.Image(rgb_j.astype(np.uint8))
        depth_i = o3d.geometry.Image(depth_i_m.astype(np.float32))
        depth_j = o3d.geometry.Image(depth_j_m.astype(np.float32))

        rgbd_i = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_i,
            depth_i,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )
        rgbd_j = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_j,
            depth_j,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.intr.width,
            self.intr.height,
            self.intr.fx,
            self.intr.fy,
            self.intr.cx,
            self.intr.cy,
        )

        pcd_i = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_i, intrinsic)
        pcd_j = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_j, intrinsic)

        # Point-to-plane requires normals
        pcd_j.estimate_normals()

        reg = o3d.pipelines.registration.registration_icp(
            pcd_i,
            pcd_j,
            self.max_corr,
            T_ij_init.astype(np.float64),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iters),
        )
        return ICPResult(T_ij=np.asarray(reg.transformation), fitness=float(reg.fitness), rmse_m=float(reg.inlier_rmse))
