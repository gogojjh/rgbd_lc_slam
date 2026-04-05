from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(args.results_dir.glob("*/metrics.json")):
        seq = p.parent.name
        m = json.loads(p.read_text(encoding="utf-8"))
        ate = m["ate_trans_m"]
        rpe_t = m["rpe"]["trans_m"]
        rpe_deg = m["rpe"]["rot_deg"]
        rows.append(
            {
                "seq": seq,
                "ate_rmse": ate["rmse"],
                "ate_mean": ate["mean"],
                "ate_p90": ate["p90"],
                "rpe_t_rmse": rpe_t["rmse"],
                "rpe_deg_rmse": rpe_deg["rmse"],
                "count": ate["count"],
            }
        )

    lines = []
    lines.append("| Sequence | Frames(eval) | ATE RMSE (m) | ATE mean/p90 (m) | RPE Δ=1 trans RMSE (m) | RPE Δ=1 rot RMSE (deg) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {seq} | {count} | {ate_rmse:.4f} | {ate_mean:.4f} / {ate_p90:.4f} | {rpe_t_rmse:.4f} | {rpe_deg_rmse:.3f} |".format(
                **r
            )
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
