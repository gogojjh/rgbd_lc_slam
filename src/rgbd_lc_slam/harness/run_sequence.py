from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from rgbd_lc_slam.harness.timers import Timer
from rgbd_lc_slam.io.tum_reader import associate_by_time, load_tum_sequence
from rgbd_lc_slam.io.trajectory_io import write_tum_trajectory


def is_icl_nuim(seq_name: str) -> bool:
    """Heuristic detection for ICL-NUIM TUM-compatible sequences."""
    s = seq_name.lower()
    return ("living_room" in s) or ("office" in s and "traj" in s)


def default_intrinsics(seq_name: str):
    """Return (Open3D intrinsics, flip_y).

    Notes:
      - TUM RGB-D: commonly used intrinsics fx=517.3 fy=516.5 cx=318.6 cy=255.3
      - ICL-NUIM: official K (VaFRIC) uses fy negative:
          [481.2, 0, 319.5;
           0, -480.0, 239.5;
           0, 0, 1]
        We emulate the negative fy convention by vertically flipping the input images
        and using fy=+480.0.
    """

    if is_icl_nuim(seq_name):
        fx, fy, cx, cy = 481.2, 480.0, 319.5, 239.5
        return o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy), True

    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    return o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy), False


def load_rgb_depth(rgb_path: Path, depth_path: Path, *, flip_y: bool = False):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise FileNotFoundError(depth_path)

    # TUM/ICL (TUM-compatible PNG) depth: uint16 with scale factor 5000 -> meters
    depth_m = depth_u16.astype(np.float32) / 5000.0

    if flip_y:
        rgb = rgb[::-1, :, :].copy()
        depth_m = depth_m[::-1, :].copy()

    return rgb, depth_m


def rgbd_from_arrays(rgb: np.ndarray, depth_m: np.ndarray):
    o3d_color = o3d.geometry.Image(rgb.astype(np.uint8))
    o3d_depth = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=1.0,
        depth_trunc=4.0,
        convert_rgb_to_intensity=False,
    )
    return rgbd


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

    def build_submap() -> o3d.geometry.PointCloud:
        if not keyframe_pcds:
            return o3d.geometry.PointCloud()
        pcd = o3d.geometry.PointCloud()
        for p in keyframe_pcds[-args.submap_k :]:
            pcd += p
        if len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(args.voxel)
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2.5, max_nn=30)
            )
        return pcd

    def should_add_keyframe(T_prev: np.ndarray, T_cur: np.ndarray) -> bool:
        dT = np.linalg.inv(T_prev) @ T_cur
        trans = np.linalg.norm(dT[:3, 3])
        # compute rotation angle
        R = dT[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
        angle_deg = float(angle * 180.0 / np.pi)
        return (trans > args.keyframe_trans) or (angle_deg > args.keyframe_rot_deg)

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

        submap = build_submap()
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

        if should_add_keyframe(keyframe_Twc[-1], Twc):
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
