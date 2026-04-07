#!/usr/bin/env bash
set -euo pipefail

# Minimal-cost single-sequence test:
# - run pose-graph backend (iSAM2) on one TUM seq
# - optionally enable loop closure
# - evaluate with SE(3) alignment (NO scale correction)
# - use higher point cloud resolution via smaller voxel

ROOT="${1:-data/tum_rgbd}"
SEQ_NAME="${2:-rgbd_dataset_freiburg1_xyz}"
OUT_RUN_DIR="${3:-results/runs/minval/${SEQ_NAME}_pgloop_se3_voxel002}"

MAX_FRAMES="${MAX_FRAMES:-100000000}"
VOXEL="${VOXEL:-0.02}"
DEVICE="${DEVICE:-cpu}"

# Loop closure knobs (can override via env)
ENABLE_LOOP="${ENABLE_LOOP:-1}"
EXCLUDE_RECENT="${EXCLUDE_RECENT:-15}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-20}"
RETRIEVAL_MIN_SCORE="${RETRIEVAL_MIN_SCORE:-0.70}"
MIN_INLIER_RATIO="${MIN_INLIER_RATIO:-0.35}"
LOOP_EVERY_KF="${LOOP_EVERY_KF:-2}"

seq_dir="$ROOT/$SEQ_NAME"
run_dir="$OUT_RUN_DIR"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

mkdir -p "$run_dir"

echo "[check] $seq_dir"
if [[ ! -d "$seq_dir" ]]; then
  echo "ERROR: missing seq dir: $seq_dir" >&2
  exit 2
fi
for f in groundtruth.txt rgb.txt depth.txt; do
  if [[ ! -f "$seq_dir/$f" ]]; then
    echo "ERROR: missing $seq_dir/$f" >&2
    exit 2
  fi
done
for dd in rgb depth; do
  if [[ ! -d "$seq_dir/$dd" ]]; then
    echo "ERROR: missing dir $seq_dir/$dd" >&2
    exit 2
  fi
done

# Run PGO pipeline (tracking + pose-graph + optional loop)
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

echo "[run] $SEQ_NAME -> $run_dir"
"${cmd[@]}"

# Copy GT into run dir for evaluation
cp "$seq_dir/groundtruth.txt" "$run_dir/traj_gt_tum.txt"

# SE(3) evaluation: align only (NO --correct_scale)

echo "[evo] SE3 tracking"
conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
  --align -r trans_part \
  --save_results "$run_dir/evo_ape_trans_se3_track.zip" \
  --plot --save_plot "$run_dir/ate_trans_se3_track.png" --no_warnings >/dev/null

conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
  --align -r trans_part --delta 1 --delta_unit f \
  --save_results "$run_dir/evo_rpe_trans_se3_track.zip" \
  --plot --save_plot "$run_dir/rpe_trans_se3_track.png" --no_warnings >/dev/null

if [[ -f "$run_dir/traj_est_pg_tum.txt" ]]; then
  echo "[evo] SE3 PGO"
  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align -r trans_part \
    --save_results "$run_dir/evo_ape_trans_se3_pg.zip" \
    --plot --save_plot "$run_dir/ate_trans_se3_pg.png" --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align -r trans_part --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_trans_se3_pg.zip" \
    --plot --save_plot "$run_dir/rpe_trans_se3_pg.png" --no_warnings >/dev/null
else
  echo "WARN: missing $run_dir/traj_est_pg_tum.txt (PGO traj)" >&2
fi

echo "Done. Results in: $run_dir"
ls -la "$run_dir" | sed -n '1,120p'
