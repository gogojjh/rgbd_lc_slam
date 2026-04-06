from __future__ import annotations

import argparse

from rgbd_lc_slam.harness.run_sequence import main


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--submap_k", type=int, default=5)
    ap.add_argument("--keyframe_trans", type=float, default=0.15)
    ap.add_argument("--keyframe_rot_deg", type=float, default=10.0)
    return ap


def cli_main() -> None:
    # Keep compatibility: delegate to harness main().
    main()


if __name__ == "__main__":
    cli_main()
