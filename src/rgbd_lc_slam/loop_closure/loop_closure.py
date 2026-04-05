from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .icp_refine import ICPRefiner
from .matching import SuperPointLightGlueMatcher
from .netvlad import NetVLADDescriptor, NetVLADWeights
from .pose_estimation import RGBDPoseEstimator, make_information_matrix
from .retrieval_db import RetrievalDatabase, RetrievalDatabaseConfig
from .types import CameraIntrinsics, LoopClosureConfig, LoopConstraint, RetrievalCandidate
from .utils import rot_angle_deg


@dataclass
class FrameData:
    frame_id: int
    rgb: np.ndarray
    depth_m: np.ndarray


class LoopClosureModule:
    """End-to-end loop closure detection for RGB-D SLAM.

    Typical usage:
        lc = LoopClosureModule(intr, LoopClosureConfig(), netvlad_weights_path=...)
        lc.add_frame(frame_id, rgb, depth)
        constraint = lc.detect_loop(frame_id, rgb, depth)

    You can also call `process_frame(...)` to add + detect in one step.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        cfg: Optional[LoopClosureConfig] = None,
        *,
        netvlad_weights_path: Optional[str] = None,
        use_faiss: bool = True,
        device: str = "cuda",
    ):
        self.intr = intrinsics
        self.cfg = cfg or LoopClosureConfig()

        self.netvlad = NetVLADDescriptor(
            device=device,
            out_dim=self.cfg.descriptor_dim,
            weights=NetVLADWeights(path=netvlad_weights_path),
        )
        self.db = RetrievalDatabase(
            RetrievalDatabaseConfig(dim=self.cfg.descriptor_dim, use_faiss=use_faiss, metric="ip")
        )
        self.matcher = SuperPointLightGlueMatcher(
            device=device,
            max_num_keypoints=self.cfg.max_num_keypoints,
        )
        self.pose_est = RGBDPoseEstimator(intrinsics, method=self.cfg.pose_method)
        self.icp = (
            ICPRefiner(
                intrinsics,
                max_corr_dist_m=self.cfg.icp_max_corr_dist_m,
                max_iters=self.cfg.icp_max_iters,
            )
            if self.cfg.use_icp
            else None
        )

        self._frames: Dict[int, FrameData] = {}

    def add_frame(self, frame_id: int, rgb: np.ndarray, depth_m: np.ndarray) -> None:
        desc = self.netvlad.compute(rgb)
        self.db.add(frame_id, desc)
        self._frames[int(frame_id)] = FrameData(frame_id=int(frame_id), rgb=rgb, depth_m=depth_m)

    def _exclude_recent_ids(self, frame_id: int) -> List[int]:
        # Exclude self and last N ids (assuming consecutive ids)
        start = frame_id - self.cfg.exclude_recent
        return [i for i in range(start, frame_id + 1) if i in self._frames]

    def detect_loop(self, frame_id: int, rgb: np.ndarray, depth_m: np.ndarray) -> Optional[LoopConstraint]:
        """Detect a loop closure for the given frame against the DB.

        Does NOT automatically add the frame to the DB.
        """
        if len(self.db) == 0:
            return None

        desc = self.netvlad.compute(rgb)
        candidates = self.db.query(
            desc,
            top_k=self.cfg.retrieval_top_k,
            exclude_ids=self._exclude_recent_ids(frame_id),
        )

        tried = 0
        for c in candidates:
            if c.score < self.cfg.retrieval_min_score:
                continue
            ref = self._frames.get(c.frame_id, None)
            if ref is None:
                continue

            tried += 1
            constraint = self._verify_and_build(
                i=c.frame_id,
                j=frame_id,
                rgb_i=ref.rgb,
                depth_i=ref.depth_m,
                rgb_j=rgb,
                depth_j=depth_m,
                retrieval_score=c.score,
            )
            if constraint is not None:
                return constraint

            if tried >= self.cfg.max_verify_per_frame:
                break

        return None

    def process_frame(self, frame_id: int, rgb: np.ndarray, depth_m: np.ndarray) -> Optional[LoopConstraint]:
        """Detect loop closure, then add frame to DB."""
        constraint = self.detect_loop(frame_id, rgb, depth_m)
        self.add_frame(frame_id, rgb, depth_m)
        return constraint

    def _verify_and_build(
        self,
        *,
        i: int,
        j: int,
        rgb_i: np.ndarray,
        depth_i: np.ndarray,
        rgb_j: np.ndarray,
        depth_j: np.ndarray,
        retrieval_score: float,
    ) -> Optional[LoopConstraint]:
        # 1) local feature matching
        m = self.matcher.match(rgb_i, rgb_j)
        if m.kpts0.shape[0] < self.cfg.min_matches:
            return None

        # 2) relative pose estimation
        est = self.pose_est.estimate(
            m.kpts0,
            m.kpts1,
            depth_i,
            depth_j,
            ransac_iters=self.cfg.ransac_iters,
            inlier_thresh_m=self.cfg.ransac_inlier_thresh_m,
            min_inliers=self.cfg.min_inliers,
        )
        if est is None:
            return None

        T_ij = est.T_ij
        num_inliers = int(est.inliers.sum())
        rmse = float(est.rmse_m)

        # 3) gating / sanity checks
        t = T_ij[:3, 3]
        trans = float(np.linalg.norm(t))
        rot_deg = rot_angle_deg(T_ij[:3, :3])

        if trans > self.cfg.max_translation_m:
            return None
        if rot_deg > self.cfg.max_rotation_deg:
            return None
        if np.isfinite(rmse) and rmse > self.cfg.max_rmse_m:
            return None

        # 4) optional ICP refinement
        if self.icp is not None:
            icp_res = self.icp.refine(rgb_i, depth_i, rgb_j, depth_j, T_ij)
            if icp_res is not None:
                # Replace with ICP if it improves fitness / rmse (simple policy)
                if icp_res.fitness > 0.2:
                    T_ij = icp_res.T_ij
                    rmse = icp_res.rmse_m

        information = make_information_matrix(num_inliers=num_inliers, rmse_m=rmse)

        return LoopConstraint(
            i=int(i),
            j=int(j),
            T_ij=T_ij.astype(np.float64),
            information=information,
            score=float(retrieval_score),
            num_inliers=num_inliers,
            rmse_m=float(rmse),
        )
