#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/tum_rgbd}"
OUT_RUNS_DIR="${2:-results/runs}"  # will create per-seq subdirs
MAX_FRAMES="${MAX_FRAMES:-100000000}"

# Avoid matplotlib trying to show windows on headless servers
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Map split names (frX/name) -> official tgz + dir
# NOTE: These URLs are untrusted external data; reviewed/verified by curl -I before use.
declare -A URLS
URLS["fr1/desk"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
URLS["fr1/room"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
URLS["fr1/xyz"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
URLS["fr2/desk"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz"
URLS["fr2/xyz"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz"
URLS["fr3/long_office_household"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
URLS["fr3/sitting_static"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz"

# split -> extracted folder name
# (official tgz contains a top-level folder with the same name as file base)
declare -A DIRS
DIRS["fr1/desk"]="rgbd_dataset_freiburg1_desk"
DIRS["fr1/room"]="rgbd_dataset_freiburg1_room"
DIRS["fr1/xyz"]="rgbd_dataset_freiburg1_xyz"
DIRS["fr2/desk"]="rgbd_dataset_freiburg2_desk"
DIRS["fr2/xyz"]="rgbd_dataset_freiburg2_xyz"
DIRS["fr3/long_office_household"]="rgbd_dataset_freiburg3_long_office_household"
DIRS["fr3/sitting_static"]="rgbd_dataset_freiburg3_sitting_static"

mkdir -p "$ROOT" "$OUT_RUNS_DIR"

SEQS=$(python - <<'PY'
import yaml
from pathlib import Path
p=Path('configs/tum_splits.yaml')
obj=yaml.safe_load(p.read_text())
seqs=obj['train']+obj['test']
print('\n'.join(seqs))
PY
)

for seq in $SEQS; do
  url="${URLS[$seq]}"
  dname="${DIRS[$seq]}"
  tgz="$ROOT/${dname}.tgz"
  extract_dir="$ROOT/$dname"

  echo "[download] $seq -> $tgz"
  if [[ ! -f "$tgz" ]]; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$tgz" "$url"
  else
    echo "  exists: $tgz"
  fi

  echo "[extract] $seq -> $extract_dir"
  if [[ ! -d "$extract_dir" ]]; then
    tar -xzf "$tgz" -C "$ROOT"
  else
    echo "  exists: $extract_dir"
  fi

done

# Run baseline tracking + evo for each sequence
for seq in $SEQS; do
  dname="${DIRS[$seq]}"
  seq_dir="$ROOT/$dname"
  run_dir="$OUT_RUNS_DIR/${seq//\//_}_full"
  mkdir -p "$run_dir"

  echo "[run] $seq_dir -> $run_dir"
  conda run -n rgbd_lc_slam python -m rgbd_lc_slam.harness.run_sequence \
    --seq_dir "$seq_dir" \
    --out_dir "$run_dir" \
    --max_frames "$MAX_FRAMES"

  echo "[evo] $seq"
  cp "$seq_dir/groundtruth.txt" "$run_dir/traj_gt_tum.txt"

  conda run -n rgbd_lc_slam evo_ape tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r full \
    --save_results "$run_dir/evo_ape.zip" \
    --plot --save_plot "$run_dir/ate_plot.png" --no_warnings >/dev/null

  conda run -n rgbd_lc_slam evo_rpe tum "$run_dir/traj_gt_tum.txt" "$run_dir/traj_est_tum.txt" \
    --align --correct_scale -r full --delta 1 --delta_unit f \
    --save_results "$run_dir/evo_rpe.zip" \
    --plot --save_plot "$run_dir/rpe_plot.png" --no_warnings >/dev/null

done

echo "All done. Results in: $OUT_RUNS_DIR"
