from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from rgbd_lc_slam.backend import ISAM2BackendConfig, PoseGraphISAM2Backend
from rgbd_lc_slam.harness.common_rgbd import default_intrinsics, load_rgb_depth, rgbd_from_arrays
from rgbd_lc_slam.harness.common_submap import build_submap, should_add_keyframe
from rgbd_lc_slam.harness.timers import Timer
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

    args = ap.parse_args()

    seq = load_tum_sequence(args.seq_dir)
    pairs = associate_by_time(seq.rgb_list, seq.depth_list, max_dt=0.02)
    pairs = pairs[: args.max_frames]

    intrinsic_o3d, flip_y = default_intrinsics(args.seq_dir.name)
    intr_lc = _intrinsics_for_loop(args.seq_dir.name, flip_y)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stamps: list[float] = []
    Twc_tracking: list[np.ndarray] = []

    # Local submap (same as baseline)
    keyframe_pcds: list[o3d.geometry.PointCloud] = []
    keyframe_Twc: list[np.ndarray] = []

    # Init pose
    Twc = np.eye(4)

    t_track = Timer()
    t_backend = Timer()
    t_loop = Timer()

    # Setup backend
    backend = PoseGraphISAM2Backend(ISAM2BackendConfig())

    # Setup loop module
    lc = None
    if args.enable_loop:
        cfg = LoopClosureConfig(
            retrieval_top_k=args.retrieval_top_k,
            retrieval_min_score=args.retrieval_min_score,
            exclude_recent=args.exclude_recent,
        )
        lc = LoopClosureModule(
            intr_lc,
            cfg,
            netvlad_weights_path=args.netvlad_weights,
            use_faiss=True,
            device=args.device,
        )

    # Seed first frame
    t0, rgb_rel, td0, depth_rel = pairs[0]
    rgb0, depth0 = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)
    rgbd0 = rgbd_from_arrays(rgb0, depth0)
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic_o3d)
    pcd0 = pcd0.voxel_down_sample(args.voxel)
    pcd0.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2.5, max_nn=30))
    keyframe_pcds.append(pcd0)
    keyframe_Twc.append(Twc.copy())

    stamps.append(float(t0))
    Twc_tracking.append(Twc.copy())

    # prior at node 0
    t_backend.tic()
    backend.add_prior(0, Twc)
    t_backend.toc()

    if lc is not None:
        # add first frame to DB
        t_loop.tic()
        lc.add_frame(0, rgb0, depth0)
        t_loop.toc()

    num_loops = 0

    for fid, (t_rgb, rgb_rel, t_d, depth_rel) in enumerate(pairs[1:], start=1):
        rgb, depth = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)
        rgbd = rgbd_from_arrays(rgb, depth)
        src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)
        src = src.voxel_down_sample(args.voxel)
        if len(src.points) == 0:
            continue
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2.5, max_nn=30))

        submap = build_submap(keyframe_pcds, submap_k=args.submap_k, voxel=args.voxel)
        if len(submap.points) == 0:
            stamps.append(float(t_rgb))
            Twc_tracking.append(Twc.copy())
            continue

        init = Twc.copy()

        t_track.tic()
        result = o3d.pipelines.registration.registration_icp(
            src,
            submap,
            max_correspondence_distance=args.voxel * 2.5,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        t_track.toc()

        Twc_prev = Twc
        Twc = result.transformation

        stamps.append(float(t_rgb))
        Twc_tracking.append(Twc.copy())

        # Add keyframe to map
        if should_add_keyframe(
            keyframe_Twc[-1],
            Twc,
            keyframe_trans=args.keyframe_trans,
            keyframe_rot_deg=args.keyframe_rot_deg,
        ):
            src_w = src.transform(Twc)
            keyframe_pcds.append(src_w)
            keyframe_Twc.append(Twc.copy())

        # Odometry factor: between = inv(Twc_prev) @ Twc
        T_odom = np.linalg.inv(Twc_prev) @ Twc
        t_backend.tic()
        backend.add_between(fid - 1, fid, T_odom, initial_Twc_j=Twc)
        t_backend.toc()

        # Loop closure factor
        if lc is not None:
            t_loop.tic()
            c = lc.process_frame(fid, rgb, depth)
            t_loop.toc()
            if c is not None:
                # LoopConstraint.T_ij maps points from frame i coordinates to frame j coordinates:
                #   p_j = T_ij @ p_i
                # Our backend uses Pose3 variable Twc (camera->world). GTSAM BetweenFactor enforces
                #   between(Twc_i, Twc_j) = inv(Twc_i) @ Twc_j
                # which maps points from frame j to frame i coordinates. Therefore we must invert.
                backend.add_between(
                    c.i,
                    c.j,
                    np.linalg.inv(c.T_ij),
                    information=c.information,
                    initial_Twc_j=Twc,
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

    timing = {
        "tracking_ms": {
            "count": t_track.stats().count,
            "p50": t_track.stats().p50_ms,
            "p90": t_track.stats().p90_ms,
            "p99": t_track.stats().p99_ms,
            "mean": t_track.stats().mean_ms,
            "max": t_track.stats().max_ms,
        },
        "backend_ms": {
            "count": t_backend.stats().count,
            "p50": t_backend.stats().p50_ms,
            "p90": t_backend.stats().p90_ms,
            "p99": t_backend.stats().p99_ms,
            "mean": t_backend.stats().mean_ms,
            "max": t_backend.stats().max_ms,
        },
        "loop_ms": {
            "count": t_loop.stats().count,
            "p50": t_loop.stats().p50_ms,
            "p90": t_loop.stats().p90_ms,
            "p99": t_loop.stats().p99_ms,
            "mean": t_loop.stats().mean_ms,
            "max": t_loop.stats().max_ms,
        },
        "num_loops": int(num_loops),
    }
    (out_dir / "timing_pg.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    print("wrote:", traj_track)
    print("wrote:", traj_pg)
    print("timing:", timing)


if __name__ == "__main__":
    main()
