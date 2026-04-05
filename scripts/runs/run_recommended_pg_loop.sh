#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/tum_rgbd}"
OUT_ROOT="${2:-results/runs/tum_pg_loop}"

mkdir -p "$OUT_ROOT"

# ---- robust logging (avoid SIGPIPE when parent process ends) ----
# Always log to file; optionally mirror to stdout only if explicitly requested.
LOG_FILE="$OUT_ROOT/run.log"
exec >>"$LOG_FILE" 2>&1

echo "[run_pg_loop] START $(date -Iseconds) ROOT=$ROOT OUT_ROOT=$OUT_ROOT"

# ---- ABI stability guard (cv2 vs torch/libstdc++) ----
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1:${LD_PRELOAD:-}"
fi

# Avoid matplotlib trying to show windows on headless servers
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Which sequences to run (recommended set)
SEQS=(
  "rgbd_dataset_freiburg3_long_office_household"
  "rgbd_dataset_freiburg1_room"
  "rgbd_dataset_freiburg2_xyz"
)

for dname in "${SEQS[@]}"; do
  seq_dir="$ROOT/$dname"
  run_dir="$OUT_ROOT/${dname}_pg_loop"
  mkdir -p "$run_dir"

  echo "[run_pg_loop] $(date -Iseconds) $seq_dir -> $run_dir"

  # Run sequence
  python -u -m rgbd_lc_slam.harness.run_sequence_pg \
    --seq_dir "$seq_dir" \
    --out_dir "$run_dir" \
    --max_frames 100000000 \
    --enable_loop \
    --device "${DEVICE:-cpu}" \
    --exclude_recent "${EXCLUDE_RECENT:-30}" \
    --retrieval_top_k "${TOP_K:-10}" \
    --retrieval_min_score "${MIN_SCORE:-0.75}" \
    ${NETVLAD_WEIGHTS:+--netvlad_weights "$NETVLAD_WEIGHTS"}

  echo "[evo] $(date -Iseconds) $dname"
  cp "$seq_dir/groundtruth.txt" "$run_dir/traj_gt_tum.txt"

  evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r full \
    --save_results "$run_dir/evo_ape_raw.zip" \
    --plot --save_plot "$run_dir/ate_plot_raw.png" --no_warnings >/dev/null

  evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r full \
    --save_results "$run_dir/evo_ape_pg.zip" \
    --plot --save_plot "$run_dir/ate_plot_pg.png" --no_warnings >/dev/null

  evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r full --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_raw.zip" \
    --plot --save_plot "$run_dir/rpe_plot_raw.png" --no_warnings >/dev/null

  evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_pg_tum.txt" \
    --align --correct_scale -r full --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe_pg.zip" \
    --plot --save_plot "$run_dir/rpe_plot_pg.png" --no_warnings >/dev/null

done

echo "[run_pg_loop] DONE $(date -Iseconds) Results in: $OUT_ROOT"
