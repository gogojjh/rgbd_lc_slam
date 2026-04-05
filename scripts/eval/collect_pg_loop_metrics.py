from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Optional


def _rmse_from_evo_zip(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with zipfile.ZipFile(path, "r") as z:
        if "stats.json" not in z.namelist():
            return None
        stats = json.loads(z.read("stats.json").decode("utf-8"))
        rmse = stats.get("rmse", None)
        return float(rmse) if rmse is not None else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for run in sorted([p for p in args.runs_dir.iterdir() if p.is_dir()]):
        seq = run.name
        ape_raw = _rmse_from_evo_zip(run / "evo_ape_raw.zip")
        rpe_raw = _rmse_from_evo_zip(run / "evo_rpe_raw.zip")
        ape_pg = _rmse_from_evo_zip(run / "evo_ape_pg.zip")
        rpe_pg = _rmse_from_evo_zip(run / "evo_rpe_pg.zip")

        timing = _read_json(run / "timing_pg.json")
        num_loops = timing.get("num_loops", None)
        loop_mean_ms = (timing.get("loop_ms", {}) or {}).get("mean", None)

        def _fmt(x: Optional[float]) -> str:
            return "" if x is None else f"{x:.4f}"

        def _impr(a: Optional[float], b: Optional[float]) -> str:
            # improvement: raw -> pg (positive means better)
            if a is None or b is None:
                return ""
            return f"{(a - b):.4f}"

        rows.append(
            {
                "seq": seq,
                "ape_raw": ape_raw,
                "ape_pg": ape_pg,
                "ape_impr": (ape_raw - ape_pg) if (ape_raw is not None and ape_pg is not None) else None,
                "rpe_raw": rpe_raw,
                "rpe_pg": rpe_pg,
                "rpe_impr": (rpe_raw - rpe_pg) if (rpe_raw is not None and rpe_pg is not None) else None,
                "num_loops": num_loops,
                "loop_mean_ms": loop_mean_ms,
            }
        )

    # CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seq",
                "ape_raw_rmse",
                "ape_pg_rmse",
                "ape_improvement",
                "rpe_raw_rmse",
                "rpe_pg_rmse",
                "rpe_improvement",
                "num_loops",
                "loop_mean_ms",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["seq"],
                    "" if r["ape_raw"] is None else f"{r['ape_raw']:.6f}",
                    "" if r["ape_pg"] is None else f"{r['ape_pg']:.6f}",
                    "" if r["ape_impr"] is None else f"{r['ape_impr']:.6f}",
                    "" if r["rpe_raw"] is None else f"{r['rpe_raw']:.6f}",
                    "" if r["rpe_pg"] is None else f"{r['rpe_pg']:.6f}",
                    "" if r["rpe_impr"] is None else f"{r['rpe_impr']:.6f}",
                    r["num_loops"] if r["num_loops"] is not None else "",
                    "" if r["loop_mean_ms"] is None else f"{float(r['loop_mean_ms']):.2f}",
                ]
            )

    # Markdown
    lines = []
    lines.append(f"# Loop-on Pose-Graph Evaluation\n")
    lines.append(f"Runs dir: `{args.runs_dir}`\n")
    lines.append(
        "| seq | APE raw | APE pg | ΔAPE (raw-pg) | RPE raw | RPE pg | ΔRPE | #loops | loop mean ms |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        def f4(x: Optional[float]) -> str:
            return "" if x is None else f"{x:.4f}"

        lines.append(
            "| {seq} | {ape_raw} | {ape_pg} | {ape_impr} | {rpe_raw} | {rpe_pg} | {rpe_impr} | {nloops} | {loopms} |".format(
                seq=r["seq"],
                ape_raw=f4(r["ape_raw"]),
                ape_pg=f4(r["ape_pg"]),
                ape_impr=f4(r["ape_impr"]),
                rpe_raw=f4(r["rpe_raw"]),
                rpe_pg=f4(r["rpe_pg"]),
                rpe_impr=f4(r["rpe_impr"]),
                nloops=r["num_loops"] if r["num_loops"] is not None else "",
                loopms="" if r["loop_mean_ms"] is None else f"{float(r['loop_mean_ms']):.0f}",
            )
        )

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("wrote:", args.out_csv)
    print("wrote:", args.out_md)


if __name__ == "__main__":
    main()
