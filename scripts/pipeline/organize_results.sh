#!/usr/bin/env bash
set -euo pipefail

# This script is a *lightweight* helper to move/copy results into a stable layout:
# results/<dataset_name>/... while keeping raw run folders under results/runs/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNS_DIR="${1:-results/runs}"
OUT_DIR="${2:-results}"

mkdir -p "$OUT_DIR/tum" "$OUT_DIR/icl"

# This repository currently keeps the authoritative raw outputs in results/runs/.
# Here we just mirror the latest summary + key figures to a stable dataset folder.

if [[ -f "$OUT_DIR/summary_tum_icl_loop_pgo.md" ]]; then
  cp -f "$OUT_DIR/summary_tum_icl_loop_pgo.md" "$OUT_DIR/tum/summary.md" || true
  cp -f "$OUT_DIR/summary_tum_icl_loop_pgo.csv" "$OUT_DIR/tum/summary.csv" || true
  cp -f "$OUT_DIR/summary_tum_icl_loop_pgo.md" "$OUT_DIR/icl/summary.md" || true
  cp -f "$OUT_DIR/summary_tum_icl_loop_pgo.csv" "$OUT_DIR/icl/summary.csv" || true
fi

echo "[organize_results] mirrored summaries to results/tum and results/icl"
