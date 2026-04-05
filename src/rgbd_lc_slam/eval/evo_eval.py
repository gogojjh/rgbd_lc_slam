from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_tum_traj(path: Path):
    # timestamp tx ty tz qx qy qz qw
    ts = []
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        t = float(parts[0])
        p = np.array([float(x) for x in parts[1:]], dtype=np.float64)
        ts.append(t)
        poses.append(p)
    return np.array(ts), np.vstack(poses) if poses else np.zeros((0, 7))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", type=Path, required=True, help="estimated traj (TUM format)")
    ap.add_argument("--gt", type=Path, required=True, help="gt traj (TUM format)")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # We rely on evo CLI for actual metric computation & plots.
    # This wrapper is a placeholder; will be extended to call evo_ape/evo_rpe and store outputs.
    print("TODO: call evo_ape/evo_rpe to compute ATE/RPE and save plots")
    print("est:", args.est)
    print("gt:", args.gt)
    print("out:", args.out_dir)


if __name__ == "__main__":
    main()
