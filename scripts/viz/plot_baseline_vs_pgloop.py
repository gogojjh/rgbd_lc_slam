from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar_compare(df: pd.DataFrame, dataset: str, metric_raw: str, metric_pg: str, ax):
    d = df[df["dataset"] == dataset].copy()
    d = d.sort_values("seq")
    labels = d["seq"].tolist()
    raw = d[metric_raw].to_numpy(float)
    pg = d[metric_pg].to_numpy(float)

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, raw, width=w, label="baseline (raw)")
    ax.bar(x + w/2, pg, width=w, label="PG+loop")
    ax.set_title(dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylabel("ATE RMSE (m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metric", type=str, default="ATE")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # keep only rows with both
    df = df.dropna(subset=["ATE_raw_rmse", "ATE_loop_pgo_rmse"])  # for ATE plot

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=180)
    bar_compare(df, "TUM", "ATE_raw_rmse", "ATE_loop_pgo_rmse", axes[0])
    bar_compare(df, "ICL", "ATE_raw_rmse", "ATE_loop_pgo_rmse", axes[1])
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Baseline vs PG+loop (ATE RMSE)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
