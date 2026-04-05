from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def write_tum_trajectory(path: Path, stamps: list[float], Twc_list: list[np.ndarray]) -> None:
    """Write trajectory in TUM format: t tx ty tz qx qy qz qw.

    Twc: 4x4 (world_T_cam)
    """
    assert len(stamps) == len(Twc_list)
    lines = []
    for t, Twc in zip(stamps, Twc_list):
        q = R.from_matrix(Twc[:3, :3]).as_quat()  # x,y,z,w
        tx, ty, tz = Twc[:3, 3]
        lines.append(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
