#!/usr/bin/env bash
set -euo pipefail

# Download + extract + run + eval all ICL-NUIM TUM-compatible sequences.
# Uses curl with resume.

ROOT="/Titan/code/robohike_ws/src/rgbd_lc_slam"
DATA_BASE="$ROOT/data/icl_nuim"
TAR_DIR="$DATA_BASE/tars"
SEQ_DIR="$DATA_BASE/sequences"
GT_DIR="$DATA_BASE/gt"
OUT_BASE="$ROOT/outputs/icl_all"

mkdir -p "$TAR_DIR" "$SEQ_DIR" "$GT_DIR" "$OUT_BASE"

# (name url gt_file)
SEQS=(
  "living_room_traj0_frei_png https://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz livingRoom0.gt.freiburg"
  "living_room_traj1_frei_png https://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz livingRoom1.gt.freiburg"
  "living_room_traj2_frei_png https://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz livingRoom2.gt.freiburg"
  "living_room_traj3_frei_png https://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz livingRoom3.gt.freiburg"
  "office_traj0_frei_png https://www.doc.ic.ac.uk/~ahanda/traj0_frei_png.tar.gz traj0.gt.freiburg"
  "office_traj1_frei_png https://www.doc.ic.ac.uk/~ahanda/traj1_frei_png.tar.gz traj1.gt.freiburg"
  "office_traj2_frei_png https://www.doc.ic.ac.uk/~ahanda/traj2_frei_png.tar.gz traj2.gt.freiburg"
  "office_traj3_frei_png https://www.doc.ic.ac.uk/~ahanda/traj3_frei_png.tar.gz traj3.gt.freiburg"
)

# Ensure GT exists (small files)
GT_URL_BASE="https://www.doc.ic.ac.uk/~ahanda/VaFRIC"
for gt in livingRoom0.gt.freiburg livingRoom1.gt.freiburg livingRoom2.gt.freiburg livingRoom3.gt.freiburg traj0.gt.freiburg traj1.gt.freiburg traj2.gt.freiburg traj3.gt.freiburg; do
  if [[ ! -f "$GT_DIR/$gt" ]]; then
    echo "[GT] downloading $gt"
    curl -L --fail -o "$GT_DIR/$gt" "$GT_URL_BASE/$gt"
  fi
done

echo "[INFO] Start processing ${#SEQS[@]} sequences"

for item in "${SEQS[@]}"; do
  name=$(echo "$item" | awk '{print $1}')
  url=$(echo "$item" | awk '{print $2}')
  gt=$(echo "$item" | awk '{print $3}')

  tar_path="$TAR_DIR/${name}.tar.gz"
  seq_path="$SEQ_DIR/$name"
  out_dir="$OUT_BASE/$name"

  echo "\n===== $name ====="

  if [[ ! -f "$tar_path" ]]; then
    echo "[DL] $url"
    curl -L --fail -C - -o "$tar_path" "$url"
  else
    echo "[DL] exists: $tar_path"
  fi

  if [[ ! -f "$seq_path/associations.txt" ]]; then
    echo "[EXTRACT] to $seq_path"
    mkdir -p "$seq_path"
    tar -xzf "$tar_path" -C "$seq_path"
  else
    echo "[EXTRACT] exists: $seq_path"
  fi

  if [[ ! -f "$seq_path/groundtruth.txt" ]]; then
    echo "[GT] copy $gt -> groundtruth.txt"
    cp "$GT_DIR/$gt" "$seq_path/groundtruth.txt"
  fi

  # Determine frame count from associations
  nframes=$(grep -v '^#' "$seq_path/associations.txt" | wc -l | tr -d ' ')
  echo "[SEQ] associations lines: $nframes"

  # Run (max_frames set to nframes)
  echo "[RUN]"
  conda run -n rgbd_lc_slam python -m rgbd_lc_slam.harness.run_sequence \
    --seq_dir "$seq_path" \
    --out_dir "$out_dir" \
    --max_frames "$nframes" \
    --voxel 0.05

  # Eval metrics
  echo "[EVAL]"
  conda run -n rgbd_lc_slam python -m rgbd_lc_slam.eval.metrics_np \
    --est "$out_dir/traj_est_tum.txt" \
    --gt "$seq_path/groundtruth.txt" \
    --max_dt 0.02 \
    --delta 1 \
    --out_json "$out_dir/metrics.json"

done

# Summarize markdown table
conda run -n rgbd_lc_slam python "$ROOT/scripts/summarize_icl_results.py" \
  --results_dir "$OUT_BASE" \
  --out_md "$ROOT/results/icl_nuim_summary.md"

echo "\n[DONE] Summary: $ROOT/results/icl_nuim_summary.md"