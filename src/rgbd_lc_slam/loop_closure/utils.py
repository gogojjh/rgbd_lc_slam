from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def to_homogeneous(T_R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = T_R
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_angle_deg(R: np.ndarray) -> float:
    # robust trace->angle
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(math.degrees(math.acos(c)))


def project_depth_to_3d(
    u: np.ndarray,
    v: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Back-project pixels with depth to 3D points (camera coords).

    Args:
        u, v: pixel coordinates (float arrays of shape (N,))
        depth_m: depth in meters (shape (N,))

    Returns:
        xyz: (N, 3)
    """
    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float64)


def gather_depth(
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    invalid_val: float = 0.0,
) -> np.ndarray:
    """Nearest-neighbor depth lookup."""
    h, w = depth.shape[:2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.full((u.shape[0],), invalid_val, dtype=depth.dtype)
    out[valid] = depth[vi[valid], ui[valid]]
    return out
