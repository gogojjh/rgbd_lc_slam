#!/usr/bin/env bash
set -euo pipefail

ROOT="/Titan/code/robohike_ws/src/rgbd_lc_slam"
SEQ_BASE="$ROOT/data/icl_nuim/sequences"
OUT_BASE="$ROOT/results/runs"

TAG="icl_pg_loop_$(date -u +%Y%m%dT%H%M%SZ)"
RUNS_DIR="$OUT_BASE/$TAG"
mkdir -p "$RUNS_DIR"

echo "[INFO] runs_dir=$RUNS_DIR"

# Loop closure params (override via env)
DEVICE="${DEVICE:-cuda}"
EXCLUDE_RECENT="${EXCLUDE_RECENT:-50}"
TOP_K="${TOP_K:-5}"
MIN_SCORE="${MIN_SCORE:-0.97}"

VOXEL="${VOXEL:-0.05}"

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
  run_dir="$RUNS_DIR/$name"
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

  echo "[RUN] loop+PGO"
  conda run -n rgbd_lc_slam python -m rgbd_lc_slam.harness.run_sequence_pg \
    --seq_dir "$seq_dir" \
    --out_dir "$run_dir" \
    --max_frames "$nframes" \
    --voxel "$VOXEL" \
    --enable_loop \
    --device "$DEVICE" \
    --exclude_recent "$EXCLUDE_RECENT" \
    --retrieval_top_k "$TOP_K" \
    --retrieval_min_score "$MIN_SCORE"

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
    --align --correct_scale -r rot_part --delta 1 --delta_unit f  \
    --save_results "$run_dir/evo_rpe_raw_rot.zip" \
    --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r trans_part \
    --save_results "$run_dir/evo_ape_pg.zip" \
    --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r rot_part \
    --save_results "$run_dir/evo_ape_pg_rot.zip" \
    --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r trans_part --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_pg.zip" \
    --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r rot_part --delta 1 --delta_unit f  \
    --save_results "$run_dir/evo_rpe_pg_rot.zip" \
    --no_warnings >/dev/null

done

echo "\n[DONE] $RUNS_DIR"
