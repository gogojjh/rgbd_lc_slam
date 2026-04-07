#!/usr/bin/env bash
set -euo pipefail

# Run ICL-NUIM baseline tracking (no loops) + evo metrics.
#
# Usage:
#   bash scripts/runs/run_all_icl_nuim_evo_baseline.sh [ICL_ROOT] [OUT_RUNS_DIR]
#
# ICL_ROOT should contain:
#   sequences/<seq>/{associations.txt,groundtruth.txt}

ROOT="/Titan/code/robohike_ws/src/rgbd_lc_slam"

ICL_ROOT="${1:-$ROOT/data/icl_nuim}"
OUT_RUNS_DIR="${2:-$ROOT/results/runs/icl_baseline_evo}"

SEQ_BASE="$ICL_ROOT/sequences"
mkdir -p "$OUT_RUNS_DIR"

echo "[INFO] icl_root=$ICL_ROOT"
echo "[INFO] out_runs_dir=$OUT_RUNS_DIR"

# Avoid matplotlib trying to show windows on headless servers
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

SEQS=(
  living_room_traj0_frei_png
  living_room_traj1_frei_png
  living_room_traj2_frei_png
  living_room_traj3_frei_png
  office_traj0_frei_png
  office_traj1_frei_png
  office_traj2_frei_png
  office_traj3_frei_png
)

for name in "${SEQS[@]}"; do
  seq_dir="$SEQ_BASE/$name"
  run_dir="$OUT_RUNS_DIR/$name"
  mkdir -p "$run_dir"

  if [[ ! -f "$seq_dir/associations.txt" ]]; then
    echo "[ERR] missing associations: $seq_dir"
    continue
  fi
  if [[ ! -f "$seq_dir/groundtruth.txt" ]]; then
    echo "[ERR] missing groundtruth: $seq_dir/groundtruth.txt"
    continue
  fi

  nframes=$(grep -v '^#' "$seq_dir/associations.txt" | wc -l | tr -d ' ')
  echo "\n===== $name (frames=$nframes) ====="

  echo "[RUN] baseline"
  conda run -n rgbd_lc_slam python -m rgbd_lc_slam.harness.run_sequence \
    --seq_dir "$seq_dir" \
    --out_dir "$run_dir" \
    --max_frames "$nframes" \
    --voxel "${VOXEL:-0.05}"

  echo "[EVO]"
  cp "$seq_dir/groundtruth.txt" "$run_dir/traj_gt_tum.txt"

  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r trans_part \
    --save_results "$run_dir/evo_ape_raw.zip" \
    --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r rot_part \
    --save_results "$run_dir/evo_ape_raw_rot.zip" \
    --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r trans_part --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_raw.zip" \
    --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r angle_deg --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_raw_rot.zip" \
    --no_warnings >/dev/null

done

echo "\n[DONE] $OUT_RUNS_DIR"
