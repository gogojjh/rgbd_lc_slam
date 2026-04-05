from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_tum_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TUM trajectory format: t tx ty tz qx qy qz qw.
    Return (t[N], xyz[N,3]).
    """
    ts = []
    xyz = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        ts.append(float(parts[0]))
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not ts:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.asarray(ts, dtype=np.float64), np.asarray(xyz, dtype=np.float64)


def associate_nearest(t_est: np.ndarray, t_gt: np.ndarray, max_dt: float) -> list[tuple[int, int]]:
    """Return list of (i_est, i_gt) nearest-neighbor associations."""
    if len(t_est) == 0 or len(t_gt) == 0:
        return []
    out = []
    for i, te in enumerate(t_est):
        j = int(np.argmin(np.abs(t_gt - te)))
        if abs(t_gt[j] - te) <= max_dt:
            out.append((i, j))
    return out


def umeyama_align(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align X to Y with a rigid transform (no scale).

    X, Y: (N,3)
    returns R(3,3), t(3,)
    minimizing || R X + t - Y ||.
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    S = (Xc.T @ Yc) / X.shape[0]
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = muY - R @ muX
    return R, t


def rmse(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v * v))) if len(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--max_dt", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t_est, p_est = read_tum_xyz(args.est)
    t_gt, p_gt = read_tum_xyz(args.gt)

    pairs = associate_nearest(t_est, t_gt, args.max_dt)
    if len(pairs) < 3:
        raise SystemExit(f"not enough associations: {len(pairs)}")

    X = np.stack([p_est[i] for i, _ in pairs], axis=0)
    Y = np.stack([p_gt[j] for _, j in pairs], axis=0)

    R, t = umeyama_align(X, Y)
    X_aligned = (R @ X.T).T + t

    e = np.linalg.norm(X_aligned - Y, axis=1)

    stats = {
        "count": int(len(e)),
        "rmse": rmse(e),
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "max": float(np.max(e)),
    }

    text = "\n".join([f"{k}: {v}" for k, v in stats.items()]) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
