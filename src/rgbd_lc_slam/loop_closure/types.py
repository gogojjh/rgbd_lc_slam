from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics.

    Parameters are in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class LoopClosureConfig:
    # Retrieval
    descriptor_dim: int = 4096
    retrieval_top_k: int = 10
    retrieval_min_score: float = 0.75  # cosine similarity / inner-product on L2-normed desc
    # If top-1 and top-2 retrieval scores are too close, the match is ambiguous.
    # Require (top1 - top2) >= margin to proceed to geometric verification.
    retrieval_min_score_margin: float = 0.0
    exclude_recent: int = 30  # ignore matches to last N frames

    # Verification budget (speed)
    max_verify_per_frame: int = 3

    # Matching
    max_num_keypoints: int = 2048
    min_matches: int = 60

    # Geometric verification
    pose_method: Literal["3d3d", "pnp"] = "3d3d"
    ransac_iters: int = 2000
    ransac_inlier_thresh_m: float = 0.05
    min_inliers: int = 40
    # Extra quality gate: require enough inliers relative to raw matches.
    min_inlier_ratio: float = 0.0
    max_rmse_m: float = 0.05

    # ICP refinement
    use_icp: bool = False
    icp_max_corr_dist_m: float = 0.07
    icp_max_iters: int = 30

    # Gating
    max_translation_m: float = 5.0
    max_rotation_deg: float = 60.0


@dataclass(frozen=True)
class RetrievalCandidate:
    frame_id: int
    score: float


@dataclass(frozen=True)
class LoopConstraint:
    """Loop closure constraint between frames i and j.

    T_ij maps a point from frame i coordinates into frame j coordinates:
        p_j = T_ij @ [p_i; 1]

    information is a 6x6 matrix in the tangent space order [rx, ry, rz, tx, ty, tz].
    """

    i: int
    j: int
    T_ij: np.ndarray  # (4,4)
    information: np.ndarray  # (6,6)
    score: float
    num_inliers: int
    rmse_m: float

    def as_tuple(self) -> Tuple[int, int, np.ndarray, np.ndarray]:
        return self.i, self.j, self.T_ij, self.information
