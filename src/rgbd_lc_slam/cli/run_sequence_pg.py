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

    # Loop closure
    ap.add_argument("--enable_loop", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--netvlad_weights", type=str, default=None)
    ap.add_argument("--exclude_recent", type=int, default=15)
    ap.add_argument("--retrieval_top_k", type=int, default=20)
    ap.add_argument("--retrieval_min_score", type=float, default=0.75)
    ap.add_argument("--retrieval_min_score_margin", type=float, default=0.0)
    ap.add_argument("--min_inlier_ratio", type=float, default=0.0)
    ap.add_argument("--loop_every_kf", type=int, default=2)

    # Backend robust kernel for loops
    ap.add_argument(
        "--robust_loop_factors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply robust kernel (Huber/Cauchy) to loop closure factors.",
    )
    ap.add_argument("--robust_loop_kernel", type=str, default="huber")
    ap.add_argument("--robust_loop_param", type=float, default=1.0)
    return ap


def cli_main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    import sys

    argv = [sys.argv[0]]
    argv += ["--seq_dir", args.seq_dir]
    argv += ["--out_dir", args.out_dir]
    argv += ["--max_frames", str(args.max_frames)]
    argv += ["--voxel", str(args.voxel)]
    argv += ["--submap_k", str(args.submap_k)]
    argv += ["--keyframe_trans", str(args.keyframe_trans)]
    argv += ["--keyframe_rot_deg", str(args.keyframe_rot_deg)]

    if args.enable_loop:
        argv += ["--enable_loop"]
    argv += ["--device", str(args.device)]
    if args.netvlad_weights is not None:
        argv += ["--netvlad_weights", str(args.netvlad_weights)]

    argv += ["--exclude_recent", str(args.exclude_recent)]
    argv += ["--retrieval_top_k", str(args.retrieval_top_k)]
    argv += ["--retrieval_min_score", str(args.retrieval_min_score)]
    argv += ["--retrieval_min_score_margin", str(args.retrieval_min_score_margin)]
    argv += ["--min_inlier_ratio", str(args.min_inlier_ratio)]
    argv += ["--loop_every_kf", str(args.loop_every_kf)]

    if args.robust_loop_factors:
        argv += ["--robust_loop_factors"]
    argv += ["--robust_loop_kernel", str(args.robust_loop_kernel)]
    argv += ["--robust_loop_param", str(args.robust_loop_param)]

    sys.argv = argv
    main()


if __name__ == "__main__":
    cli_main()
