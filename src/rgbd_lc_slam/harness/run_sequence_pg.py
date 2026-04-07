from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgbd_lc_slam.backend import ISAM2BackendConfig, PoseGraphISAM2Backend
from rgbd_lc_slam.frontend import RGBDFrame, RGBDICPFrontend, RGBDTrackingConfig
from rgbd_lc_slam.io.rgbd_io import default_intrinsics, load_rgb_depth
from rgbd_lc_slam.io.tum_reader import associate_by_time, load_tum_sequence
from rgbd_lc_slam.io.trajectory_io import write_tum_trajectory
from rgbd_lc_slam.loop_closure import CameraIntrinsics, LoopClosureConfig, LoopClosureModule


def _intrinsics_for_loop(seq_name: str, flip_y: bool) -> CameraIntrinsics:
    # keep consistent with frontend intrinsics used by point cloud generation
    if flip_y:
        # ICL-NUIM, see default_intrinsics(): we flip images vertically.
        fx, fy, cx, cy = 481.2, 480.0, 319.5, 239.5
    else:
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--submap_k", type=int, default=5)
    ap.add_argument("--keyframe_trans", type=float, default=0.15)
    ap.add_argument("--keyframe_rot_deg", type=float, default=10.0)

    # Loop closure
    ap.add_argument("--enable_loop", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--netvlad_weights", type=str, default=None)
    ap.add_argument("--exclude_recent", type=int, default=30)
    ap.add_argument("--retrieval_top_k", type=int, default=10)
    ap.add_argument("--retrieval_min_score", type=float, default=0.75)
    ap.add_argument(
        "--retrieval_min_score_margin",
        type=float,
        default=0.0,
        help="Require (top1 - top2) >= margin to verify (Phase2).",
    )
    ap.add_argument("--min_inlier_ratio", type=float, default=0.0)
    ap.add_argument(
        "--loop_every_kf",
        type=int,
        default=2,
        help="Run loop detection every K keyframes (Phase1 throttling).",
    )

    # Backend robust kernel for loops
    ap.add_argument(
        "--robust_loop_factors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply robust kernel (Huber/Cauchy) to loop closure factors.",
    )
    ap.add_argument(
        "--robust_loop_kernel",
        type=str,
        default="huber",
        help="Robust kernel type for loop factors: huber | cauchy.",
    )
    ap.add_argument(
        "--robust_loop_param",
        type=float,
        default=1.0,
        help="Robust kernel parameter k (GTSAM mEstimator).",
    )

    args = ap.parse_args()

    seq = load_tum_sequence(args.seq_dir)
    pairs = associate_by_time(seq.rgb_list, seq.depth_list, max_dt=0.02)
    pairs = pairs[: args.max_frames]

    intrinsic_o3d, flip_y = default_intrinsics(args.seq_dir.name)
    intr_lc = _intrinsics_for_loop(args.seq_dir.name, flip_y)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup frontend
    cfg = RGBDTrackingConfig(
        voxel=args.voxel,
        submap_k=args.submap_k,
        keyframe_trans=args.keyframe_trans,
        keyframe_rot_deg=args.keyframe_rot_deg,
    )
    fe = RGBDICPFrontend(intrinsic=intrinsic_o3d, cfg=cfg)

    stamps: list[float] = []
    Twc_tracking: list[np.ndarray] = []
    tracking_ms_list: list[float] = []

    t_backend_ms: list[float] = []
    t_loop_ms: list[float] = []

    # Setup backend
    backend_cfg = ISAM2BackendConfig(
        robust_loop_factors=bool(args.robust_loop_factors),
        robust_loop_kernel=str(args.robust_loop_kernel),
        robust_loop_param=float(args.robust_loop_param),
    )
    backend = PoseGraphISAM2Backend(backend_cfg)

    # Setup loop module
    lc = None
    if args.enable_loop:
        cfg_lc = LoopClosureConfig(
            retrieval_top_k=args.retrieval_top_k,
            retrieval_min_score=args.retrieval_min_score,
            retrieval_min_score_margin=args.retrieval_min_score_margin,
            exclude_recent=args.exclude_recent,
            min_inlier_ratio=args.min_inlier_ratio,
        )
        lc = LoopClosureModule(
            intr_lc,
            cfg_lc,
            netvlad_weights_path=args.netvlad_weights,
            use_faiss=True,
            device=args.device,
        )

    # Seed first frame
    t0, rgb_rel, _, depth_rel = pairs[0]
    rgb0, depth0 = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)
    r0 = fe.seed(RGBDFrame(fid=0, stamp=float(t0), rgb=rgb0, depth_m=depth0))

    stamps.append(r0.stamp)
    Twc_tracking.append(r0.Twc)
    tracking_ms_list.append(r0.tracking_ms)

    # prior at node 0
    import time

    t0b = time.perf_counter()
    backend.add_prior(0, r0.Twc)
    t_backend_ms.append((time.perf_counter() - t0b) * 1000.0)

    if lc is not None:
        t0l = time.perf_counter()
        lc.add_frame(0, rgb0, depth0)
        t_loop_ms.append((time.perf_counter() - t0l) * 1000.0)

    num_loops = 0
    kf_count = 1  # fid=0 is a keyframe by construction
    last_loop_kf_count = -10**9

    for fid, (t_rgb, rgb_rel, _, depth_rel) in enumerate(pairs[1:], start=1):
        rgb, depth = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)

        Twc_prev = Twc_tracking[-1]
        r = fe.track(RGBDFrame(fid=fid, stamp=float(t_rgb), rgb=rgb, depth_m=depth))

        if r.is_keyframe:
            kf_count += 1

        stamps.append(r.stamp)
        Twc_tracking.append(r.Twc)
        tracking_ms_list.append(r.tracking_ms)

        # Odometry factor: between = inv(Twc_prev) @ Twc
        T_odom = np.linalg.inv(Twc_prev) @ r.Twc
        t0b = time.perf_counter()
        backend.add_between(fid - 1, fid, T_odom, initial_Twc_j=r.Twc)
        t_backend_ms.append((time.perf_counter() - t0b) * 1000.0)

        # Loop closure factor
        if lc is not None:
            t0l = time.perf_counter()

            # Phase1: only attempt loop closure on keyframes, and only every K keyframes.
            do_loop = bool(r.is_keyframe) and (kf_count - last_loop_kf_count >= int(args.loop_every_kf))
            c = None
            if do_loop:
                c = lc.process_frame(fid, rgb, depth)
                last_loop_kf_count = kf_count
            else:
                lc.add_frame(fid, rgb, depth)

            t_loop_ms.append((time.perf_counter() - t0l) * 1000.0)
            if c is not None:
                backend.add_between(
                    c.i,
                    c.j,
                    np.linalg.inv(c.T_ij),
                    information=c.information,
                    initial_Twc_j=r.Twc,
                )
                num_loops += 1

    # Export trajectories
    traj_track = out_dir / "traj_est_tum.txt"
    write_tum_trajectory(traj_track, stamps, Twc_tracking)

    Twc_opt = backend.calculate_estimate_Twc()
    stamps_opt, Twc_opt_list = [], []
    for fid, t in enumerate(stamps):
        if fid in Twc_opt:
            stamps_opt.append(float(t))
            Twc_opt_list.append(Twc_opt[fid])

    traj_pg = out_dir / "traj_est_pg_tum.txt"
    write_tum_trajectory(traj_pg, stamps_opt, Twc_opt_list)

    def _stats(arr: list[float]):
        if len(arr) == 0:
            return {"count": 0}
        a = np.asarray(arr, dtype=np.float64)
        return {
            "count": int(a.size),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()),
            "max": float(a.max()),
        }

    timing = {
        "tracking_ms": _stats(tracking_ms_list),
        "backend_ms": _stats(t_backend_ms),
        "loop_ms": _stats(t_loop_ms),
        "num_loops": int(num_loops),
    }
    (out_dir / "timing_pg.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    # Phase0: loop diagnostics
    if lc is not None:
        (out_dir / "loop_stats.json").write_text(
            json.dumps(lc.get_diagnostics(), indent=2),
            encoding="utf-8",
        )

    print("wrote:", traj_track)
    print("wrote:", traj_pg)
    print("timing:", timing)


if __name__ == "__main__":
    main()
