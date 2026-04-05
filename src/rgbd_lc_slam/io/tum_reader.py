from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TumSequence:
    root: Path
    rgb_dir: Path
    depth_dir: Path
    rgb_list: list[tuple[float, str]]
    depth_list: list[tuple[float, str]]
    gt: list[tuple[float, np.ndarray]]  # (t, [tx,ty,tz,qx,qy,qz,qw])


def _read_assoc_list(path: Path) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t_s, rel = line.split()[:2]
        out.append((float(t_s), rel))
    return out


def _read_assoc_pairs(path: Path) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Read ICL-NUIM style associations.txt.

    Common format per line (after comments):
      t0 rel0 t1 rel1

    However, in the wild both orders exist:
      - (t_rgb rgb_rel t_depth depth_rel)
      - (t_depth depth_rel t_rgb rgb_rel)

    We detect which rel belongs to rgb/depth by substring match.
    """

    rgb_list: list[tuple[float, str]] = []
    depth_list: list[tuple[float, str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        t0_s, rel0, t1_s, rel1 = parts[:4]
        t0, t1 = float(t0_s), float(t1_s)

        rel0_l = rel0.lower()
        rel1_l = rel1.lower()

        rel0_is_rgb = ("rgb" in rel0_l) or ("color" in rel0_l)
        rel0_is_depth = "depth" in rel0_l
        rel1_is_rgb = ("rgb" in rel1_l) or ("color" in rel1_l)
        rel1_is_depth = "depth" in rel1_l

        if rel0_is_rgb and rel1_is_depth:
            rgb_list.append((t0, rel0))
            depth_list.append((t1, rel1))
        elif rel0_is_depth and rel1_is_rgb:
            rgb_list.append((t1, rel1))
            depth_list.append((t0, rel0))
        else:
            # Fallback: assume rel0=rgb, rel1=depth (original doc format)
            rgb_list.append((t0, rel0))
            depth_list.append((t1, rel1))

    return rgb_list, depth_list


def _read_gt(path: Path) -> list[tuple[float, np.ndarray]]:
    out: list[tuple[float, np.ndarray]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        t = float(parts[0])
        v = np.array([float(x) for x in parts[1:8]], dtype=np.float64)
        out.append((t, v))
    return out


def load_tum_sequence(seq_dir: Path) -> TumSequence:
    """Load a sequence.

    Supports:
      - TUM RGB-D style: rgb.txt, depth.txt, groundtruth.txt
      - ICL-NUIM style: associations.txt + groundtruth.txt (or copied/linked)

    Both are mapped into the same TumSequence structure.
    """

    seq_dir = seq_dir.expanduser().resolve()

    assoc_pairs = seq_dir / "associations.txt"
    rgb_txt = seq_dir / "rgb.txt"
    depth_txt = seq_dir / "depth.txt"

    if assoc_pairs.exists() and (not rgb_txt.exists() or not depth_txt.exists()):
        rgb_list, depth_list = _read_assoc_pairs(assoc_pairs)
    else:
        rgb_list = _read_assoc_list(rgb_txt)
        depth_list = _read_assoc_list(depth_txt)

    gt = _read_gt(seq_dir / "groundtruth.txt")
    return TumSequence(
        root=seq_dir,
        rgb_dir=seq_dir / "rgb",
        depth_dir=seq_dir / "depth",
        rgb_list=rgb_list,
        depth_list=depth_list,
        gt=gt,
    )


def associate_by_time(
    rgb_list: list[tuple[float, str]],
    depth_list: list[tuple[float, str]],
    max_dt: float = 0.02,
) -> list[tuple[float, str, float, str]]:
    """Greedy associate rgb and depth by nearest timestamp."""
    depth_ts = np.array([t for t, _ in depth_list], dtype=np.float64)
    out: list[tuple[float, str, float, str]] = []
    for t_rgb, rgb_rel in rgb_list:
        j = int(np.argmin(np.abs(depth_ts - t_rgb)))
        t_d, d_rel = depth_list[j]
        if abs(t_d - t_rgb) <= max_dt:
            out.append((t_rgb, rgb_rel, t_d, d_rel))
    return out


def interp_gt(gt: list[tuple[float, np.ndarray]], t: float) -> np.ndarray | None:
    """Return nearest GT (no interpolation yet) for smoke test."""
    if not gt:
        return None
    ts = np.array([x[0] for x in gt], dtype=np.float64)
    i = int(np.argmin(np.abs(ts - t)))
    return gt[i][1]
