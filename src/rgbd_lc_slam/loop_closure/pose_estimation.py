from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .types import CameraIntrinsics
from .utils import gather_depth, project_depth_to_3d, rot_angle_deg, to_homogeneous


def umeyama_alignment(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate rigid transform (R,t) aligning X to Y.

    Minimizes sum ||R X_i + t - Y_i||^2.

    Args:
        X: (N,3) source
        Y: (N,3) target

    Returns:
        R: (3,3)
        t: (3,)
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    Sigma = (Yc.T @ Xc) / n
    U, _, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = mu_Y - R @ mu_X
    return R.astype(np.float64), t.astype(np.float64)


def ransac_rigid(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    iters: int = 2000,
    inlier_thresh_m: float = 0.05,
    min_inliers: int = 40,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """RANSAC for rigid alignment with Umeyama.

    Returns:
        T: (4,4) or None
        inliers: (N,) bool
        rmse_m: float on inliers (inf if failure)
    """
    if rng is None:
        rng = np.random.default_rng()
    N = X.shape[0]
    if N < 3:
        return None, np.zeros((N,), dtype=bool), float("inf")

    best_inliers = np.zeros((N,), dtype=bool)
    best_num = 0
    best_T = None
    best_rmse = float("inf")

    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        R, t = umeyama_alignment(X[idx], Y[idx])
        pred = (R @ X.T).T + t
        err = np.linalg.norm(pred - Y, axis=1)
        inliers = err < inlier_thresh_m
        num = int(inliers.sum())
        if num < best_num:
            continue
        if num >= 3:
            # re-fit on inliers
            R2, t2 = umeyama_alignment(X[inliers], Y[inliers])
            pred2 = (R2 @ X[inliers].T).T + t2
            rmse = float(np.sqrt(np.mean(np.sum((pred2 - Y[inliers]) ** 2, axis=1))))
        else:
            rmse = float("inf")
            R2, t2 = R, t

        if num > best_num or (num == best_num and rmse < best_rmse):
            best_num = num
            best_inliers = inliers
            best_T = to_homogeneous(R2, t2)
            best_rmse = rmse

    if best_T is None or best_num < min_inliers:
        return None, best_inliers, float("inf")

    return best_T, best_inliers, best_rmse


@dataclass
class PoseEstimate:
    T_ij: np.ndarray
    inliers: np.ndarray
    rmse_m: float


class RGBDPoseEstimator:
    """Estimate relative pose from depth-assisted correspondences."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        method: Literal["3d3d", "pnp"] = "3d3d",
    ):
        self.K = intrinsics.K()
        self.intr = intrinsics
        self.method = method

    def estimate(
        self,
        kpts_i: np.ndarray,
        kpts_j: np.ndarray,
        depth_i_m: np.ndarray,
        depth_j_m: np.ndarray,
        *,
        ransac_iters: int = 2000,
        inlier_thresh_m: float = 0.05,
        min_inliers: int = 40,
    ) -> Optional[PoseEstimate]:
        """Estimate T_ij from frame i to frame j."""

        if self.method == "3d3d":
            return self._estimate_3d3d(
                kpts_i,
                kpts_j,
                depth_i_m,
                depth_j_m,
                ransac_iters=ransac_iters,
                inlier_thresh_m=inlier_thresh_m,
                min_inliers=min_inliers,
            )
        else:
            return self._estimate_pnp(
                kpts_i,
                kpts_j,
                depth_i_m,
                ransac_iters=ransac_iters,
                min_inliers=min_inliers,
            )

    def _estimate_3d3d(
        self,
        kpts_i: np.ndarray,
        kpts_j: np.ndarray,
        depth_i_m: np.ndarray,
        depth_j_m: np.ndarray,
        *,
        ransac_iters: int,
        inlier_thresh_m: float,
        min_inliers: int,
    ) -> Optional[PoseEstimate]:
        u_i, v_i = kpts_i[:, 0], kpts_i[:, 1]
        u_j, v_j = kpts_j[:, 0], kpts_j[:, 1]

        di = gather_depth(depth_i_m, u_i, v_i, invalid_val=0.0).astype(np.float64)
        dj = gather_depth(depth_j_m, u_j, v_j, invalid_val=0.0).astype(np.float64)

        valid = (di > 0.1) & (dj > 0.1) & np.isfinite(di) & np.isfinite(dj)
        if int(valid.sum()) < 3:
            return None

        Xi = project_depth_to_3d(
            u_i[valid],
            v_i[valid],
            di[valid],
            self.intr.fx,
            self.intr.fy,
            self.intr.cx,
            self.intr.cy,
        )
        Xj = project_depth_to_3d(
            u_j[valid],
            v_j[valid],
            dj[valid],
            self.intr.fx,
            self.intr.fy,
            self.intr.cx,
            self.intr.cy,
        )

        T, inl, rmse = ransac_rigid(
            Xi,
            Xj,
            iters=ransac_iters,
            inlier_thresh_m=inlier_thresh_m,
            min_inliers=min_inliers,
        )
        if T is None:
            return None
        return PoseEstimate(T_ij=T, inliers=inl, rmse_m=rmse)

    def _estimate_pnp(
        self,
        kpts_i: np.ndarray,
        kpts_j: np.ndarray,
        depth_i_m: np.ndarray,
        *,
        ransac_iters: int,
        min_inliers: int,
    ) -> Optional[PoseEstimate]:
        # 3D from i + 2D in j
        import cv2

        u_i, v_i = kpts_i[:, 0], kpts_i[:, 1]
        di = gather_depth(depth_i_m, u_i, v_i, invalid_val=0.0).astype(np.float64)
        valid = (di > 0.1) & np.isfinite(di)
        if int(valid.sum()) < 6:
            return None

        Xi = project_depth_to_3d(
            u_i[valid],
            v_i[valid],
            di[valid],
            self.intr.fx,
            self.intr.fy,
            self.intr.cx,
            self.intr.cy,
        ).astype(np.float64)
        xj = kpts_j[valid].astype(np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=Xi,
            imagePoints=xj,
            cameraMatrix=self.K,
            distCoeffs=None,
            iterationsCount=ransac_iters,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < min_inliers:
            return None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        T = to_homogeneous(R, t)

        # approximate RMSE in 3D by backprojecting inliers through depth? (placeholder)
        rmse = float("nan")
        inl_mask = np.zeros((int(valid.sum()),), dtype=bool)
        inl_mask[inliers.reshape(-1)] = True
        return PoseEstimate(T_ij=T, inliers=inl_mask, rmse_m=rmse)


def make_information_matrix(num_inliers: int, rmse_m: float) -> np.ndarray:
    """Heuristic information matrix from inliers and residual.

    For a real system, replace with a noise model based on sensor properties.
    """
    rmse = float(rmse_m if np.isfinite(rmse_m) else 0.05)
    rmse = max(rmse, 1e-3)

    # stronger constraint with more inliers and lower rmse
    w = float(max(1.0, min(1e4, num_inliers / (rmse**2))))

    # simple diagonal: rotation (rad) and translation (m)
    sigma_t = max(0.02, rmse)
    sigma_r = max(0.5 * math.radians(1.0), rmse)  # very rough

    info = np.diag(
        [w / (sigma_r**2)] * 3 + [w / (sigma_t**2)] * 3
    ).astype(np.float64)
    return info
