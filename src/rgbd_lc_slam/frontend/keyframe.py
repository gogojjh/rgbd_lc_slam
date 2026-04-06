from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rotation_angle_deg(R: np.ndarray) -> float:
    # Robust angle from trace
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(tr)))


def should_add_keyframe(Twc_last_kf: np.ndarray, Twc: np.ndarray, *, keyframe_trans: float, keyframe_rot_deg: float) -> bool:
    d = Twc_last_kf[:3, 3] - Twc[:3, 3]
    dist = float(np.linalg.norm(d))

    R_rel = Twc_last_kf[:3, :3].T @ Twc[:3, :3]
    ang = _rotation_angle_deg(R_rel)

    return (dist > float(keyframe_trans)) or (ang > float(keyframe_rot_deg))


@dataclass
class PoseBuffer:
    """Keeps recent poses for reference."""

    Twc_last: np.ndarray
    Twc_last_kf: np.ndarray
