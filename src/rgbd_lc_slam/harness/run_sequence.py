from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from rgbd_lc_slam.harness.common_rgbd import default_intrinsics, load_rgb_depth, rgbd_from_arrays
from rgbd_lc_slam.harness.common_submap import build_submap, should_add_keyframe
from rgbd_lc_slam.harness.timers import Timer
from rgbd_lc_slam.io.tum_reader import associate_by_time, load_tum_sequence
from rgbd_lc_slam.io.trajectory_io import write_tum_trajectory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--submap_k", type=int, default=5, help="#recent keyframes to build local submap")
    ap.add_argument("--keyframe_trans", type=float, default=0.15)
    ap.add_argument("--keyframe_rot_deg", type=float, default=10.0)
    args = ap.parse_args()

    seq = load_tum_sequence(args.seq_dir)
    pairs = associate_by_time(seq.rgb_list, seq.depth_list, max_dt=0.02)
    pairs = pairs[: args.max_frames]

    intrinsic, flip_y = default_intrinsics(args.seq_dir.name)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    Twc_list: list[np.ndarray] = []
    stamps: list[float] = []

    # Local submap: store recent keyframe pointclouds in world frame
    keyframe_pcds: list[o3d.geometry.PointCloud] = []
    keyframe_Twc: list[np.ndarray] = []

    # Init pose
    Twc = np.eye(4)
    prev_Twc = Twc.copy()

    t_track = Timer()

    # Seed first frame as keyframe
    t0, rgb_rel, td0, depth_rel = pairs[0]
    rgb0, depth0 = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)
    rgbd0 = rgbd_from_arrays(rgb0, depth0)
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic)
    pcd0 = pcd0.voxel_down_sample(args.voxel)
    pcd0.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2.5, max_nn=30))
    keyframe_pcds.append(pcd0)
    keyframe_Twc.append(Twc.copy())

    Twc_list.append(Twc.copy())
    stamps.append(float(t0))

    for (t_rgb, rgb_rel, t_d, depth_rel) in pairs[1:]:
        rgb, depth = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)
        rgbd = rgbd_from_arrays(rgb, depth)
        src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        src = src.voxel_down_sample(args.voxel)
        if len(src.points) == 0:
            continue
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2.5, max_nn=30))

        submap = build_submap(keyframe_pcds, submap_k=args.submap_k, voxel=args.voxel)
        if len(submap.points) == 0:
            # no map yet
            Twc_list.append(Twc.copy())
            stamps.append(float(t_rgb))
            continue

        # map-to-frame: register src (current) to submap (map)
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
        dt_ms = t_track.toc()

        Twc = result.transformation

        Twc_list.append(Twc.copy())
        stamps.append(float(t_rgb))

        if should_add_keyframe(
            keyframe_Twc[-1],
            Twc,
            keyframe_trans=args.keyframe_trans,
            keyframe_rot_deg=args.keyframe_rot_deg,
        ):
            # transform current cloud to world and add
            src_w = src.transform(Twc)
            keyframe_pcds.append(src_w)
            keyframe_Twc.append(Twc.copy())

    traj_path = out_dir / "traj_est_tum.txt"
    write_tum_trajectory(traj_path, stamps, Twc_list)

    timing = {
        "tracking_ms": {
            "count": t_track.stats().count,
            "p50": t_track.stats().p50_ms,
            "p90": t_track.stats().p90_ms,
            "p99": t_track.stats().p99_ms,
            "mean": t_track.stats().mean_ms,
            "max": t_track.stats().max_ms,
        }
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("wrote:", traj_path)
    print("timing:", timing)


if __name__ == "__main__":
    main()
