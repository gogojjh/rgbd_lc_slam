from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


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
