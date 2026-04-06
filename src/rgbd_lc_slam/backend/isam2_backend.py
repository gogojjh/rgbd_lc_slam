from __future__ import annotations

"""GTSAM iSAM2 incremental pose-graph backend.

State variable convention:
  - We optimize Twc (Pose3): transform from camera frame to world frame.
  - Between measurement for factor (i,j) should be:
      T_ij_between = inv(Twc_i) @ Twc_j

This matches gtsam Pose3.between() convention.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _try_import_gtsam():
    try:
        import gtsam  # type: ignore

        return gtsam
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "gtsam is required for the backend. Install via `pip install gtsam`."
        ) from e


gtsam = _try_import_gtsam()


@dataclass
class ISAM2BackendConfig:
    # Prior on the first pose (Twc_0)
    prior_sigmas: Tuple[float, float, float, float, float, float] = (
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
    )  # (rot rad)*3 + (trans m)*3

    # Default odometry noise if caller doesn't provide information
    odom_sigmas: Tuple[float, float, float, float, float, float] = (
        np.deg2rad(2.0),
        np.deg2rad(2.0),
        np.deg2rad(2.0),
        0.05,
        0.05,
        0.05,
    )

    # If True, apply a robust kernel to loop-closure between factors (Phase2).
    robust_loop_factors: bool = True
    robust_loop_kernel: str = "huber"  # "huber" | "cauchy"
    robust_loop_param: float = 1.0

    # iSAM2 parameters
    relinearize_threshold: float = 0.01
    relinearize_skip: int = 1


def pose3_from_T(T: np.ndarray) -> "gtsam.Pose3":
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4), got {T.shape}")
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    return gtsam.Pose3(R, t)


def T_from_pose3(p: "gtsam.Pose3") -> np.ndarray:
    R = np.asarray(p.rotation().matrix(), dtype=np.float64)
    t = p.translation()
    if hasattr(t, "x"):
        tv = np.array([t.x(), t.y(), t.z()], dtype=np.float64)
    else:
        tv = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tv
    return T


def export_trajectory_tum(
    stamps: Iterable[float],
    Twc_by_id: Dict[int, np.ndarray],
) -> Tuple[list[float], list[np.ndarray]]:
    """Create ordered (stamps, Twc_list) for writing TUM trajectory.

    Args:
        stamps: iterable of timestamps in frame-id order (assumes frame id = index)
        Twc_by_id: dict mapping frame_id -> Twc 4x4

    Returns:
        (stamps_out, Twc_list)
    """
    stamps_out: list[float] = []
    Twc_list: list[np.ndarray] = []
    for fid, t in enumerate(stamps):
        if fid not in Twc_by_id:
            # skip missing
            continue
        stamps_out.append(float(t))
        Twc_list.append(np.asarray(Twc_by_id[fid], dtype=np.float64))
    return stamps_out, Twc_list


class PoseGraphISAM2Backend:
    """Incremental pose-graph optimizer using iSAM2."""

    def __init__(self, cfg: Optional[ISAM2BackendConfig] = None):
        self.cfg = cfg or ISAM2BackendConfig()

        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(self.cfg.relinearize_threshold)
        params.setRelinearizeSkip(int(self.cfg.relinearize_skip))
        self.isam2 = gtsam.ISAM2(params)

        self._initialized = False
        self._latest_estimate = None

        # Keep track of which nodes have been inserted
        self._have_node: set[int] = set()

    @staticmethod
    def _key(i: int) -> int:
        return gtsam.symbol("x", int(i))

    def _robustify_if_requested(self, noise, *, is_loop: bool):
        if not is_loop or not bool(self.cfg.robust_loop_factors):
            return noise

        kernel = str(self.cfg.robust_loop_kernel).lower()
        k = float(self.cfg.robust_loop_param)
        if kernel == "cauchy":
            robust = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy(k), noise)
        else:
            robust = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber(k), noise)
        return robust

    def add_prior(self, i: int, Twc: np.ndarray) -> None:
        """Add a prior factor on pose i and initialize the graph."""
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.cfg.prior_sigmas, dtype=np.float64))
        graph.add(gtsam.PriorFactorPose3(self._key(i), pose3_from_T(Twc), noise))
        values.insert(self._key(i), pose3_from_T(Twc))

        self.isam2.update(graph, values)
        self._have_node.add(int(i))
        self._initialized = True
        self._latest_estimate = self.isam2.calculateEstimate()

    def add_between(
        self,
        i: int,
        j: int,
        T_ij_between: np.ndarray,
        *,
        information: Optional[np.ndarray] = None,
        initial_Twc_j: Optional[np.ndarray] = None,
    ) -> None:
        """Add between factor i->j.

        Args:
            i, j: node ids
            T_ij_between: inv(Twc_i) @ Twc_j (4x4)
            information: 6x6 information matrix in [rx ry rz tx ty tz]
            initial_Twc_j: optional initial guess for Twc_j to insert if new.
        """
        if not self._initialized:
            raise RuntimeError("Call add_prior() before adding factors")

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        if information is None:
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.cfg.odom_sigmas, dtype=np.float64))
        else:
            info = np.asarray(information, dtype=np.float64)
            if info.shape != (6, 6):
                raise ValueError(f"Expected information (6,6), got {info.shape}")
            noise = gtsam.noiseModel.Gaussian.Information(info)

        is_loop = abs(int(j) - int(i)) > 1
        noise = self._robustify_if_requested(noise, is_loop=is_loop)

        graph.add(
            gtsam.BetweenFactorPose3(
                self._key(i),
                self._key(j),
                pose3_from_T(T_ij_between),
                noise,
            )
        )

        if int(j) not in self._have_node:
            if initial_Twc_j is None:
                # default initial guess: propagate from i
                # Twc_j ≈ Twc_i * T_ij
                if self._latest_estimate is None or not self._latest_estimate.exists(self._key(i)):
                    raise RuntimeError("Missing estimate for node i; provide initial_Twc_j")
                Twc_i = T_from_pose3(self._latest_estimate.atPose3(self._key(i)))
                Twc_j0 = Twc_i @ np.asarray(T_ij_between, dtype=np.float64)
            else:
                Twc_j0 = np.asarray(initial_Twc_j, dtype=np.float64)
            values.insert(self._key(j), pose3_from_T(Twc_j0))
            self._have_node.add(int(j))

        self.isam2.update(graph, values)
        self._latest_estimate = self.isam2.calculateEstimate()

    def calculate_estimate_Twc(self) -> Dict[int, np.ndarray]:
        if self._latest_estimate is None:
            return {}
        out: Dict[int, np.ndarray] = {}
        for i in sorted(self._have_node):
            k = self._key(i)
            if self._latest_estimate.exists(k):
                out[int(i)] = T_from_pose3(self._latest_estimate.atPose3(k))
        return out
