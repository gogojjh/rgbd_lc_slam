#!/usr/bin/env bash
set -euo pipefail

# Batch minimal-cost test on the TUM fr1 subset:
# - run pose-graph backend (iSAM2) + optional loop closure
# - higher point cloud resolution via smaller voxel
# - evaluate with SE(3) alignment (NO scale correction)
# - additionally compute Sim(3) ATE for paper comparison convenience

ROOT="${1:-data/tum_rgbd}"
OUT_RUNS_DIR="${2:-results/runs/tum_fr1_mast3r_subset_pgloop_se3_voxel002}"

MAX_FRAMES="${MAX_FRAMES:-100000000}"
VOXEL="${VOXEL:-0.02}"
DEVICE="${DEVICE:-cpu}"

ENABLE_LOOP="${ENABLE_LOOP:-1}"
EXCLUDE_RECENT="${EXCLUDE_RECENT:-15}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-20}"
RETRIEVAL_MIN_SCORE="${RETRIEVAL_MIN_SCORE:-0.70}"
MIN_INLIER_RATIO="${MIN_INLIER_RATIO:-0.35}"
LOOP_EVERY_KF="${LOOP_EVERY_KF:-2}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

SEQS=(
  rgbd_dataset_freiburg1_360
  rgbd_dataset_freiburg1_desk
  rgbd_dataset_freiburg1_room
  rgbd_dataset_freiburg1_xyz
)

mkdir -p "$OUT_RUNS_DIR"

for dname in "${SEQS[@]}"; do
  seq_dir="$ROOT/$dname"
  run_dir="$OUT_RUNS_DIR/${dname}_pgloop"

  echo "[check] $seq_dir"
  if [[ ! -d "$seq_dir" ]]; then
    echo "  SKIP: missing dir"
    continue
  fi
  for f in groundtruth.txt rgb.txt depth.txt; do
    if [[ ! -f "$seq_dir/$f" ]]; then
      echo "  SKIP: missing $f"
      continue 2
    fi
  done
  for dd in rgb depth; do
    if [[ ! -d "$seq_dir/$dd" ]]; then
      echo "  SKIP: missing dir $dd"
      continue 2
    fi
  done

  mkdir -p "$run_dir"

  cmd=(
    conda run -n rgbd_lc_slam python -m rgbd_lc_slam.harness.run_sequence_pg
    --seq_dir "$seq_dir"
    --out_dir "$run_dir"
    --max_frames "$MAX_FRAMES"
    --voxel "$VOXEL"
    --device "$DEVICE"
  )

  if [[ "$ENABLE_LOOP" == "1" ]]; then
    cmd+=(
      --enable_loop
      --exclude_recent "$EXCLUDE_RECENT"
      --retrieval_top_k "$RETRIEVAL_TOP_K"
      --retrieval_min_score "$RETRIEVAL_MIN_SCORE"
      --min_inlier_ratio "$MIN_INLIER_RATIO"
      --loop_every_kf "$LOOP_EVERY_KF"
    )
  fi

  echo "[run] $dname -> $run_dir"
  "${cmd[@]}"

  echo "[evo] $dname"
  cp "$seq_dir/groundtruth.txt" "$run_dir/traj_gt_tum.txt"

  # --- SE3 (align only) ---
  # Tracking
  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align -r trans_part \
    --save_results "$run_dir/evo_ape_raw.zip" \
    --plot --save_plot "$run_dir/ate_trans_se3_track.png" --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align -r trans_part --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_raw.zip" \
    --plot --save_plot "$run_dir/rpe_trans_se3_track.png" --no_warnings >/dev/null

  # PGO
  if [[ -f "$run_dir/traj_est_pg_tum.txt" ]]; then
    conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
      --align -r trans_part \
      --save_results "$run_dir/evo_ape_pg.zip" \
      --plot --save_plot "$run_dir/ate_trans_se3_pg.png" --no_warnings >/dev/null

    conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
      --align -r trans_part --delta 1 --delta_unit f \
      --save_results "$run_dir/evo_rpe_pg.zip" \
      --plot --save_plot "$run_dir/rpe_trans_se3_pg.png" --no_warnings >/dev/null
  fi

  # --- Sim3 (align + correct_scale) for convenience ---
  # Tracking
  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r trans_part \
    --save_results "$run_dir/evo_ape_raw_sim3.zip" \
    --no_warnings >/dev/null

  # PGO
  if [[ -f "$run_dir/traj_est_pg_tum.txt" ]]; then
    conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
      --align --correct_scale -r trans_part \
      --save_results "$run_dir/evo_ape_pg_sim3.zip" \
      --no_warnings >/dev/null
  fi

done

echo "All done. Results in: $OUT_RUNS_DIR"
