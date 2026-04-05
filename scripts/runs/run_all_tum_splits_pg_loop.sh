#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/tum_rgbd}"
OUT_RUNS_DIR="${2:-results/runs/tum_splits_pg_loop}"  # will create per-seq subdirs
MAX_FRAMES="${MAX_FRAMES:-100000000}"

# ---- ABI stability guard (cv2 vs torch/libstdc++) ----
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1:${LD_PRELOAD:-}"
fi

# Loop closure params (can override via env)
DEVICE="${DEVICE:-cpu}"
EXCLUDE_RECENT="${EXCLUDE_RECENT:-30}"
TOP_K="${TOP_K:-10}"
MIN_SCORE="${MIN_SCORE:-0.75}"
NETVLAD_WEIGHTS="${NETVLAD_WEIGHTS:-}"

mkdir -p "$OUT_RUNS_DIR"

SEQS=$(python - <<'PY'
import yaml
from pathlib import Path
p=Path('configs/tum_splits.yaml')
obj=yaml.safe_load(p.read_text())
seqs=obj['train']+obj['test']
print('\n'.join(seqs))
PY
)

# Map split names (frX/name) -> extracted folder name
declare -A DIRS
DIRS["fr1/desk"]="rgbd_dataset_freiburg1_desk"
DIRS["fr1/room"]="rgbd_dataset_freiburg1_room"
DIRS["fr1/xyz"]="rgbd_dataset_freiburg1_xyz"
DIRS["fr2/desk"]="rgbd_dataset_freiburg2_desk"
DIRS["fr2/xyz"]="rgbd_dataset_freiburg2_xyz"
DIRS["fr3/long_office_household"]="rgbd_dataset_freiburg3_long_office_household"
DIRS["fr3/sitting_static"]="rgbd_dataset_freiburg3_sitting_static"

for seq in $SEQS; do
  dname="${DIRS[$seq]}"
  seq_dir="$ROOT/$dname"
  run_dir="$OUT_RUNS_DIR/${seq//\//_}_full_pg_loop"
  mkdir -p "$run_dir"

  echo "[run_pg_loop] $seq_dir -> $run_dir"
  if [[ -n "$NETVLAD_WEIGHTS" ]]; then
    python -m rgbd_lc_slam.harness.run_sequence_pg \
      --seq_dir "$seq_dir" \
      --out_dir "$run_dir" \
      --max_frames "$MAX_FRAMES" \
      --enable_loop \
      --device "$DEVICE" \
      --exclude_recent "$EXCLUDE_RECENT" \
      --retrieval_top_k "$TOP_K" \
      --retrieval_min_score "$MIN_SCORE" \
      --netvlad_weights "$NETVLAD_WEIGHTS"
  else
    python -m rgbd_lc_slam.harness.run_sequence_pg \
      --seq_dir "$seq_dir" \
      --out_dir "$run_dir" \
      --max_frames "$MAX_FRAMES" \
      --enable_loop \
      --device "$DEVICE" \
      --exclude_recent "$EXCLUDE_RECENT" \
      --retrieval_top_k "$TOP_K" \
      --retrieval_min_score "$MIN_SCORE"
  fi

  echo "[evo] $seq"
  cp "$seq_dir/groundtruth.txt" "$run_dir/traj_gt_tum.txt"

  # Tracking (raw)
  evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r trans_part \
    --save_results "$run_dir/evo_ape_raw.zip" \
    --plot --save_plot "$run_dir/ate_plot_raw.png" >/dev/null
  evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r rot_part \
    --save_results "$run_dir/evo_ape_raw_rot.zip" \
    --plot --save_plot "$run_dir/ate_plot_raw_rot.png" >/dev/null

  evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r trans_part --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_raw.zip" \
    --plot --save_plot "$run_dir/rpe_plot_raw.png" >/dev/null
  evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r rot_part --delta 1 --delta_unit f --change_unit deg \
    --save_results "$run_dir/evo_rpe_raw_rot.zip" \
    --plot --save_plot "$run_dir/rpe_plot_raw_rot.png" >/dev/null

  # Pose-graph optimized
  evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r trans_part \
    --save_results "$run_dir/evo_ape_pg.zip" \
    --plot --save_plot "$run_dir/ate_plot_pg.png" >/dev/null
  evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r rot_part \
    --save_results "$run_dir/evo_ape_pg_rot.zip" \
    --plot --save_plot "$run_dir/ate_plot_pg_rot.png" >/dev/null

  evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r trans_part --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_pg.zip" \
    --plot --save_plot "$run_dir/rpe_plot_pg.png" >/dev/null
  evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r rot_part --delta 1 --delta_unit f --change_unit deg \
    --save_results "$run_dir/evo_rpe_pg_rot.zip" \
    --plot --save_plot "$run_dir/rpe_plot_pg_rot.png" >/dev/null

done

echo "All done. Results in: $OUT_RUNS_DIR"
