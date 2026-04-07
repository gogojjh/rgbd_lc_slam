#!/usr/bin/env bash
set -euo pipefail

# Run only the already-present TUM fr1 sequences (NO downloads).
# Outputs per-sequence run dirs with traj + evo zip metrics.

ROOT="${1:-data/tum_rgbd}"
OUT_RUNS_DIR="${2:-results/runs/tum_fr1_mast3r_subset}"
MAX_FRAMES="${MAX_FRAMES:-100000000}"

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
  run_dir="$OUT_RUNS_DIR/${dname}_raw"

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

  echo "[run] $dname -> $run_dir"
  conda run -n rgbd_lc_slam python -m rgbd_lc_slam.harness.run_sequence \
    --seq_dir "$seq_dir" \
    --out_dir "$run_dir" \
    --max_frames "$MAX_FRAMES"

  echo "[evo] $dname"
  cp "$seq_dir/groundtruth.txt" "$run_dir/traj_gt_tum.txt"

  # ATE translation RMSE (m)
  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r trans_part \
    --save_results "$run_dir/evo_ape_trans.zip" \
    --plot --save_plot "$run_dir/ate_trans_plot.png" --no_warnings >/dev/null

  # RPE translation RMSE (m), delta=1 frame
  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r trans_part --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_trans.zip" \
    --plot --save_plot "$run_dir/rpe_trans_plot.png" --no_warnings >/dev/null

done

echo "All done. Results in: $OUT_RUNS_DIR"