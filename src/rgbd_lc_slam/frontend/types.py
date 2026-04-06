from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBDTrackingConfig:
    voxel: float = 0.05
    submap_k: int = 5
    keyframe_trans: float = 0.15
    keyframe_rot_deg: float = 10.0

    # ICP
    max_corr_mult: float = 2.5
    icp_max_iter: int = 30


@dataclass(frozen=True)
class RGBDFrame:
    fid: int
    stamp: float
    rgb: np.ndarray  # uint8 HxWx3, RGB
    depth_m: np.ndarray  # float32 HxW, meters


@dataclass(frozen=True)
class TrackingResult:
    fid: int
    stamp: float
    Twc: np.ndarray  # 4x4
    is_keyframe: bool
    tracking_ms: float
