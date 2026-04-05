from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rgbd_lc_slam.eval.ate_eval_np import read_tum_xyz


def associate_nearest(t_ref: np.ndarray, t_q: np.ndarray, max_dt: float) -> list[tuple[int, int]]:
    out = []
    for i, tr in enumerate(t_ref):
        j = int(np.argmin(np.abs(t_q - tr)))
        if abs(t_q[j] - tr) <= max_dt:
            out.append((i, j))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', type=Path, required=True)
    ap.add_argument('--tracking', type=Path, required=True)
    ap.add_argument('--pgo', type=Path, required=True)
    ap.add_argument('--out_png', type=Path, required=True)
    ap.add_argument('--max_dt', type=float, default=0.02)
    ap.add_argument('--title', type=str, default='ICL trajectory comparison')
    args = ap.parse_args()

    t_gt, p_gt = read_tum_xyz(args.gt)
    t_tr, p_tr = read_tum_xyz(args.tracking)
    t_pg, p_pg = read_tum_xyz(args.pgo)

    # Align by timestamp to GT for plotting (no SE(3) align here, just time association)
    pairs_tr = associate_nearest(t_gt, t_tr, args.max_dt)
    pairs_pg = associate_nearest(t_gt, t_pg, args.max_dt)

    gt_xy = p_gt[:, [0, 2]]  # X-Z top-down

    tr_xy = np.array([p_tr[j, [0, 2]] for _, j in pairs_tr]) if pairs_tr else np.zeros((0, 2))
    gt_tr_xy = np.array([p_gt[i, [0, 2]] for i, _ in pairs_tr]) if pairs_tr else np.zeros((0, 2))

    pg_xy = np.array([p_pg[j, [0, 2]] for _, j in pairs_pg]) if pairs_pg else np.zeros((0, 2))
    gt_pg_xy = np.array([p_gt[i, [0, 2]] for i, _ in pairs_pg]) if pairs_pg else np.zeros((0, 2))

    plt.figure(figsize=(7, 6), dpi=160)
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], 'k-', linewidth=2, label='GT')
    if len(tr_xy):
        plt.plot(tr_xy[:, 0], tr_xy[:, 1], 'C1-', linewidth=1.5, label='Tracking')
    if len(pg_xy):
        plt.plot(pg_xy[:, 0], pg_xy[:, 1], 'C0-', linewidth=1.5, label='Loop+PGO')

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title(args.title)
    plt.legend()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png)
    print('wrote', args.out_png)


if __name__ == '__main__':
    main()
