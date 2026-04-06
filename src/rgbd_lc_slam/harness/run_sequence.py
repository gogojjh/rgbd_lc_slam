from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgbd_lc_slam.frontend import RGBDFrame, RGBDICPFrontend, RGBDTrackingConfig
from rgbd_lc_slam.io.rgbd_io import default_intrinsics, load_rgb_depth
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

    cfg = RGBDTrackingConfig(
        voxel=args.voxel,
        submap_k=args.submap_k,
        keyframe_trans=args.keyframe_trans,
        keyframe_rot_deg=args.keyframe_rot_deg,
    )
    fe = RGBDICPFrontend(intrinsic=intrinsic, cfg=cfg)

    Twc_list: list[np.ndarray] = []
    stamps: list[float] = []
    tracking_ms_list: list[float] = []

    # Seed
    t0, rgb_rel, _, depth_rel = pairs[0]
    rgb0, depth0 = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)
    r0 = fe.seed(RGBDFrame(fid=0, stamp=float(t0), rgb=rgb0, depth_m=depth0))
    Twc_list.append(r0.Twc)
    stamps.append(r0.stamp)
    tracking_ms_list.append(r0.tracking_ms)

    for fid, (t_rgb, rgb_rel, _, depth_rel) in enumerate(pairs[1:], start=1):
        rgb, depth = load_rgb_depth(seq.root / rgb_rel, seq.root / depth_rel, flip_y=flip_y)
        r = fe.track(RGBDFrame(fid=fid, stamp=float(t_rgb), rgb=rgb, depth_m=depth))
        Twc_list.append(r.Twc)
        stamps.append(r.stamp)
        tracking_ms_list.append(r.tracking_ms)

    traj_path = out_dir / "traj_est_tum.txt"
    write_tum_trajectory(traj_path, stamps, Twc_list)

    tracking_ms_arr = np.array(tracking_ms_list, dtype=np.float64)
    timing = {
        "tracking_ms": {
            "count": int(tracking_ms_arr.size),
            "p50": float(np.percentile(tracking_ms_arr, 50)),
            "p90": float(np.percentile(tracking_ms_arr, 90)),
            "p99": float(np.percentile(tracking_ms_arr, 99)),
            "mean": float(tracking_ms_arr.mean()),
            "max": float(tracking_ms_arr.max()),
        }
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("wrote:", traj_path)
    print("timing:", timing)


if __name__ == "__main__":
    main()
