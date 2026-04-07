from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class TumTraj:
    t: np.ndarray  # (N,)
    xyz: np.ndarray  # (N,3)


def load_tum(path: Path) -> TumTraj:
    ts: list[float] = []
    xs: list[list[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts.append(float(parts[0]))
            xs.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not xs:
        raise ValueError(f"No poses found in {path}")

    t = np.asarray(ts, dtype=float)
    xyz = np.asarray(xs, dtype=float)

    order = np.argsort(t)
    return TumTraj(t=t[order], xyz=xyz[order])


def associate_by_timestamp(
    ref: TumTraj,
    est: TumTraj,
    *,
    max_diff_s: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate ref/est by nearest timestamps (two-pointer).

    Returns (ref_xyz_matched, est_xyz_matched).
    """

    ref_i = 0
    ref_xyz: list[np.ndarray] = []
    est_xyz: list[np.ndarray] = []

    for t_est, xyz_est in zip(est.t, est.xyz, strict=False):
        while ref_i + 1 < ref.t.shape[0] and ref.t[ref_i + 1] <= t_est:
            ref_i += 1

        candidates = [ref_i]
        if ref_i + 1 < ref.t.shape[0]:
            candidates.append(ref_i + 1)

        best_j = min(candidates, key=lambda j: abs(ref.t[j] - t_est))
        if abs(ref.t[best_j] - t_est) <= max_diff_s:
            ref_xyz.append(ref.xyz[best_j])
            est_xyz.append(xyz_est)

    if len(ref_xyz) < 3:
        raise ValueError(
            f"Too few timestamp matches: {len(ref_xyz)} (max_diff_s={max_diff_s})"
        )

    return np.stack(ref_xyz, axis=0), np.stack(est_xyz, axis=0)


def rigid_align(ref_xyz: np.ndarray, est_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-align est to ref using Kabsch (no scale)."""

    ref_mean = ref_xyz.mean(axis=0)
    est_mean = est_xyz.mean(axis=0)
    ref0 = ref_xyz - ref_mean
    est0 = est_xyz - est_mean

    h = est0.T @ ref0
    u, _s, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    t = ref_mean - est_mean @ r
    return r, t


def align_est_to_ref(ref: TumTraj, est: TumTraj) -> np.ndarray:
    ref_m, est_m = associate_by_timestamp(ref, est)
    r, t = rigid_align(ref_m, est_m)
    return (est.xyz @ r) + t


def plot_dataset(
    *,
    title: str,
    items: list[tuple[str, Path, Path, Path]],
    out_path: Path,
    ncols: int = 4,
) -> None:
    n = len(items)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows))
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    else:
        axes_list = list(np.ravel(axes))

    for ax in axes_list[n:]:
        ax.axis("off")

    for idx, (name, gt_path, baseline_path, pg_path) in enumerate(items):
        ax = axes_list[idx]

        gt = load_tum(gt_path)
        baseline = load_tum(baseline_path)
        pg = load_tum(pg_path)

        baseline_aligned = align_est_to_ref(gt, baseline)
        pg_aligned = align_est_to_ref(gt, pg)

        ax.plot(gt.xyz[:, 0], gt.xyz[:, 2], color="black", linewidth=1.5, label="GT")
        ax.plot(
            baseline_aligned[:, 0],
            baseline_aligned[:, 2],
            color="#1f77b4",
            linewidth=1.2,
            label="baseline",
        )
        ax.plot(
            pg_aligned[:, 0],
            pg_aligned[:, 2],
            color="#d62728",
            linewidth=1.2,
            label="loop+PGO (after)",
        )

        ax.set_title(name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path("results/runs/phase4_fullsuite_20260407T041034Z")

    tum_baseline = root / "tum_splits_baseline"
    tum_pg = root / "tum_splits_pg_loop"
    icl_baseline = root / "icl_baseline_evo"
    icl_pg = root / "icl_pg_loop"

    tum_map = {
        "fr1_desk_full": "rgbd_dataset_freiburg1_desk",
        "fr1_room_full": "rgbd_dataset_freiburg1_room",
        "fr1_xyz_full": "rgbd_dataset_freiburg1_xyz",
        "fr2_desk_full": "rgbd_dataset_freiburg2_desk",
        "fr2_xyz_full": "rgbd_dataset_freiburg2_xyz",
        "fr3_long_office_household_full": "rgbd_dataset_freiburg3_long_office_household",
        "fr3_sitting_static_full": "rgbd_dataset_freiburg3_sitting_static",
    }

    tum_items: list[tuple[str, Path, Path, Path]] = []
    for seq, dname in tum_map.items():
        gt_path = tum_baseline / seq / "traj_gt_tum.txt"
        baseline_path = tum_baseline / seq / "traj_est_tum.txt"
        pg_path = tum_pg / f"{dname}_pg_loop" / "traj_est_pg_tum.txt"
        tum_items.append((seq, gt_path, baseline_path, pg_path))

    icl_seqs = [
        "living_room_traj0_frei_png",
        "living_room_traj1_frei_png",
        "living_room_traj2_frei_png",
        "living_room_traj3_frei_png",
        "office_traj0_frei_png",
        "office_traj1_frei_png",
        "office_traj2_frei_png",
        "office_traj3_frei_png",
    ]
    icl_items: list[tuple[str, Path, Path, Path]] = []
    for seq in icl_seqs:
        gt_path = icl_baseline / seq / "traj_gt_tum.txt"
        baseline_path = icl_baseline / seq / "traj_est_tum.txt"
        pg_path = icl_pg / seq / "traj_est_pg_tum.txt"
        icl_items.append((seq, gt_path, baseline_path, pg_path))

    out_dir = Path("results/plots")
    plot_dataset(
        title="TUM: GT vs baseline vs loop+PGO(after) (XZ projection, aligned)",
        items=tum_items,
        out_path=out_dir / "traj_compare_tum_gt_baseline_loop_pgo_after.png",
        ncols=4,
    )
    plot_dataset(
        title="ICL-NUIM: GT vs baseline vs loop+PGO(after) (XZ projection, aligned)",
        items=icl_items,
        out_path=out_dir / "traj_compare_icl_gt_baseline_loop_pgo_after.png",
        ncols=4,
    )


if __name__ == "__main__":
    main()
