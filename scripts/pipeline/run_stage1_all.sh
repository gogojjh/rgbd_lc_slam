#!/usr/bin/env bash
set -euo pipefail

# One-click runner for Stage-1 baseline & loop+PGO experiments.
# It only orchestrates existing scripts; you can override dataset roots via env.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

TUM_ROOT="${TUM_ROOT:-data/tum_rgbd}"
ICL_ROOT="${ICL_ROOT:-data/icl_nuim}"

RUNS_DIR="${RUNS_DIR:-results/runs}"

echo "[stage1] ROOT=$ROOT_DIR"
echo "[stage1] TUM_ROOT=$TUM_ROOT"
echo "[stage1] ICL_ROOT=$ICL_ROOT"
echo "[stage1] RUNS_DIR=$RUNS_DIR"

# 1) baseline
bash scripts/runs/run_all_tum_splits.sh "$TUM_ROOT" "$RUNS_DIR/tum_splits_baseline"
bash scripts/runs/run_all_icl_nuim.sh  "$ICL_ROOT" "$RUNS_DIR/icl_baseline"

# 2) loop+PGO
bash scripts/runs/run_all_tum_splits_pg_loop.sh "$TUM_ROOT" "$RUNS_DIR/tum_pg_loop"
bash scripts/runs/run_all_icl_nuim_pg_loop.sh "$ICL_ROOT" "$RUNS_DIR/icl_pg_loop"

# 3) summarize
python scripts/eval/summarize_all_results.py \
  --tum_baseline "$RUNS_DIR/tum_splits_baseline" \
  --tum_pg_loop   "$RUNS_DIR/tum_pg_loop" \
  --icl_baseline  "$RUNS_DIR/icl_baseline" \
  --icl_pg_loop   "$RUNS_DIR/icl_pg_loop" \
  --out_md results/summary_tum_icl_loop_pgo.md \
  --out_csv results/summary_tum_icl_loop_pgo.csv

echo "[stage1] DONE"
