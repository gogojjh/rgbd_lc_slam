from __future__ import annotations

import argparse

from rgbd_lc_slam.harness.run_sequence_pg import main


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--submap_k", type=int, default=5)
    ap.add_argument("--keyframe_trans", type=float, default=0.15)
    ap.add_argument("--keyframe_rot_deg", type=float, default=10.0)

    ap.add_argument("--enable_loop", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--netvlad_weights", type=str, default=None)
    ap.add_argument("--exclude_recent", type=int, default=30)
    ap.add_argument("--retrieval_top_k", type=int, default=10)
    ap.add_argument("--retrieval_min_score", type=float, default=0.75)
    return ap


def cli_main() -> None:
    main()


if __name__ == "__main__":
    cli_main()
