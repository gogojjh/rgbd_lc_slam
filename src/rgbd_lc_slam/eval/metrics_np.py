from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_tum(path: Path):
    """Read TUM trajectory format: t tx ty tz qx qy qz qw."""
    ts = []
    p = []
    q = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        ts.append(float(parts[0]))
        p.append([float(parts[1]), float(parts[2]), float(parts[3])])
        q.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not ts:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 4), dtype=np.float64),
        )
    return np.asarray(ts, np.float64), np.asarray(p, np.float64), np.asarray(q, np.float64)


def associate_nearest(t_est: np.ndarray, t_gt: np.ndarray, max_dt: float) -> list[tuple[int, int]]:
    if len(t_est) == 0 or len(t_gt) == 0:
        return []
    out: list[tuple[int, int]] = []
    for i, te in enumerate(t_est):
        j = int(np.argmin(np.abs(t_gt - te)))
        if abs(t_gt[j] - te) <= max_dt:
            out.append((i, j))
    return out


def umeyama_align(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid alignment (no scale): minimize || R X + t - Y ||."""
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    S = (Xc.T @ Yc) / X.shape[0]
    U, _, Vt = np.linalg.svd(S)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = muY - Rm @ muX
    return Rm, t


def angle_from_R(Rm: np.ndarray) -> float:
    """Return angle (rad) of rotation matrix."""
    c = (np.trace(Rm) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c))


def stats_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "count": int(x.size),
        "rmse": float(np.sqrt(np.mean(x * x))) if x.size else float("nan"),
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "median": float(np.median(x)) if x.size else float("nan"),
        "p90": float(np.percentile(x, 90)) if x.size else float("nan"),
        "max": float(np.max(x)) if x.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--max_dt", type=float, default=0.02)
    ap.add_argument("--delta", type=int, default=1, help="RPE delta in frames (after association)")
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    t_est, p_est, q_est = read_tum(args.est)
    t_gt, p_gt, q_gt = read_tum(args.gt)

    pairs = associate_nearest(t_est, t_gt, args.max_dt)
    if len(pairs) < 3:
        raise SystemExit(f"not enough associations: {len(pairs)}")

    idx_est = np.array([i for i, _ in pairs], dtype=int)
    idx_gt = np.array([j for _, j in pairs], dtype=int)

    X = p_est[idx_est]
    Y = p_gt[idx_gt]

    R_align, t_align = umeyama_align(X, Y)
    X_aligned = (R_align @ X.T).T + t_align

    ate = np.linalg.norm(X_aligned - Y, axis=1)

    # RPE (relative pose error) using associated, aligned poses
    d = int(args.delta)
    if len(pairs) <= d:
        rpe_t = np.zeros((0,), dtype=np.float64)
        rpe_r = np.zeros((0,), dtype=np.float64)
    else:
        # positions
        dp_est = X_aligned[d:] - X_aligned[:-d]
        dp_gt = Y[d:] - Y[:-d]
        rpe_t = np.linalg.norm(dp_est - dp_gt, axis=1)

        # rotations
        R_est = R.from_quat(q_est[idx_est]).as_matrix()  # (N,3,3)
        R_gt = R.from_quat(q_gt[idx_gt]).as_matrix()

        # apply global alignment rotation to estimated orientations
        R_est_aligned = R_align @ R_est

        R_rel_est = np.einsum("nij,njk->nik", np.transpose(R_est_aligned[:-d], (0, 2, 1)), R_est_aligned[d:])
        R_rel_gt = np.einsum("nij,njk->nik", np.transpose(R_gt[:-d], (0, 2, 1)), R_gt[d:])

        R_err = np.einsum("nij,njk->nik", np.transpose(R_rel_gt, (0, 2, 1)), R_rel_est)
        rpe_r = np.array([angle_from_R(M) for M in R_err], dtype=np.float64)

    out = {
        "assoc": {"count": int(len(pairs)), "max_dt": float(args.max_dt)},
        "ate_trans_m": stats_1d(ate),
        "rpe": {
            "delta": d,
            "trans_m": stats_1d(rpe_t),
            "rot_rad": stats_1d(rpe_r),
            "rot_deg": stats_1d(np.rad2deg(rpe_r)),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
