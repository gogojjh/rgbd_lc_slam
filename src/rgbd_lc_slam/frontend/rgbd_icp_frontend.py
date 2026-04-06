from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from rgbd_lc_slam.frontend.types import RGBDTrackingConfig, RGBDFrame, TrackingResult
from rgbd_lc_slam.frontend.keyframe import should_add_keyframe
from rgbd_lc_slam.frontend.submap import build_submap


@dataclass
class RGBDICPFrontend:
    """A minimal RGB-D tracking frontend.

    - Maintains a recent-keyframe submap in world frame.
    - Performs map-to-frame ICP (point-to-plane) to estimate Twc for each frame.

    This is intentionally close to the original harness logic (Milestone-2 goal: modularize
    without changing the algorithm).
    """

    intrinsic: o3d.camera.PinholeCameraIntrinsic
    cfg: RGBDTrackingConfig

    Twc: np.ndarray | None = None
    keyframe_pcds_w: list[o3d.geometry.PointCloud] | None = None
    keyframe_Twc: list[np.ndarray] | None = None

    def reset(self) -> None:
        self.Twc = np.eye(4)
        self.keyframe_pcds_w = []
        self.keyframe_Twc = []

    def _pcd_from_frame(self, frame: RGBDFrame) -> o3d.geometry.PointCloud:
        # Local import for speed only; keep helper in harness/common_rgbd
        o3d_color = o3d.geometry.Image(frame.rgb.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(frame.depth_m.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=4.0,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd = pcd.voxel_down_sample(self.cfg.voxel)
        if len(pcd.points) > 0:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.cfg.voxel * 2.5, max_nn=30)
            )
        return pcd

    def seed(self, frame0: RGBDFrame) -> TrackingResult:
        if self.Twc is None or self.keyframe_pcds_w is None or self.keyframe_Twc is None:
            self.reset()

        assert self.Twc is not None
        assert self.keyframe_pcds_w is not None
        assert self.keyframe_Twc is not None

        pcd0 = self._pcd_from_frame(frame0)
        # For the first frame, store as keyframe in *world* frame (Twc=I).
        self.keyframe_pcds_w.append(pcd0)
        self.keyframe_Twc.append(self.Twc.copy())

        return TrackingResult(
            fid=frame0.fid,
            stamp=frame0.stamp,
            Twc=self.Twc.copy(),
            is_keyframe=True,
            tracking_ms=0.0,
        )

    def track(self, frame: RGBDFrame) -> TrackingResult:
        if self.Twc is None or self.keyframe_pcds_w is None or self.keyframe_Twc is None:
            raise RuntimeError("Call seed() before track().")

        src = self._pcd_from_frame(frame)
        if len(src.points) == 0:
            return TrackingResult(
                fid=frame.fid,
                stamp=frame.stamp,
                Twc=self.Twc.copy(),
                is_keyframe=False,
                tracking_ms=0.0,
            )

        submap = build_submap(self.keyframe_pcds_w, submap_k=self.cfg.submap_k, voxel=self.cfg.voxel)
        if len(submap.points) == 0:
            return TrackingResult(
                fid=frame.fid,
                stamp=frame.stamp,
                Twc=self.Twc.copy(),
                is_keyframe=False,
                tracking_ms=0.0,
            )

        init = self.Twc.copy()

        import time

        t0 = time.perf_counter()
        result = o3d.pipelines.registration.registration_icp(
            src,
            submap,
            max_correspondence_distance=self.cfg.voxel * self.cfg.max_corr_mult,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.cfg.icp_max_iter),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.Twc = result.transformation

        is_kf = should_add_keyframe(
            self.keyframe_Twc[-1],
            self.Twc,
            keyframe_trans=self.cfg.keyframe_trans,
            keyframe_rot_deg=self.cfg.keyframe_rot_deg,
        )
        if is_kf:
            # Add current cloud in world frame (avoid in-place modifying `src`)
            src_w = o3d.geometry.PointCloud(src)
            src_w.transform(self.Twc)
            self.keyframe_pcds_w.append(src_w)
            self.keyframe_Twc.append(self.Twc.copy())

        return TrackingResult(
            fid=frame.fid,
            stamp=frame.stamp,
            Twc=self.Twc.copy(),
            is_keyframe=bool(is_kf),
            tracking_ms=float(dt_ms),
        )
