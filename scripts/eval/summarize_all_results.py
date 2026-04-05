from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional


def rmse_from_zip(path: Path) -> Optional[float]:
    if not path or (not path.exists()):
        return None
    with zipfile.ZipFile(path, "r") as z:
        if "stats.json" not in z.namelist():
            return None
        stats = json.loads(z.read("stats.json").decode("utf-8"))
    v = stats.get("rmse", None)
    return None if v is None else float(v)


def find_any(run_dir: Path, names: list[str]) -> Optional[Path]:
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tum_baseline", type=Path, required=True)
    ap.add_argument("--tum_pg_loop", type=Path, required=True)
    ap.add_argument("--icl_baseline", type=Path, required=True)
    ap.add_argument("--icl_pg_loop", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    rows = []

    def add_dataset(dataset: str, baseline_dir: Path, pgloop_dir: Path):
        # baseline: evo_ape.zip/evo_rpe.zip (or raw)
        for run in sorted([p for p in baseline_dir.iterdir() if p.is_dir()]):
            seq = run.name
            # baseline
            ape_raw_p = find_any(run, ["evo_ape.zip", "evo_ape_raw.zip"])
            rpe_raw_p = find_any(run, ["evo_rpe.zip", "evo_rpe_raw.zip"])
            # optional diagnostic (rotation)
            ape_raw_rot_p = find_any(run, ["evo_ape_raw_rot.zip", "evo_ape_rot.zip"])
            rpe_raw_rot_p = find_any(run, ["evo_rpe_raw_rot.zip", "evo_rpe_rot.zip"])

            ape_raw = rmse_from_zip(ape_raw_p) if ape_raw_p else None
            rpe_raw = rmse_from_zip(rpe_raw_p) if rpe_raw_p else None
            ape_raw_rot = rmse_from_zip(ape_raw_rot_p) if ape_raw_rot_p else None
            rpe_raw_rot = rmse_from_zip(rpe_raw_rot_p) if rpe_raw_rot_p else None

            # pg loop: try direct name, else try mapping TUM extracted folder names
            pg = pgloop_dir / seq
            if (not pg.exists()) and dataset == "TUM":
                # baseline uses frX_name_full; pgloop runs use extracted folder name
                tum_map = {
                    "fr1_desk_full": "rgbd_dataset_freiburg1_desk_pg_loop",
                    "fr1_room_full": "rgbd_dataset_freiburg1_room_pg_loop",
                    "fr1_xyz_full": "rgbd_dataset_freiburg1_xyz_pg_loop",
                    "fr2_desk_full": "rgbd_dataset_freiburg2_desk_pg_loop",
                    "fr2_xyz_full": "rgbd_dataset_freiburg2_xyz_pg_loop",
                    "fr3_long_office_household_full": "rgbd_dataset_freiburg3_long_office_household_pg_loop",
                    "fr3_sitting_static_full": "rgbd_dataset_freiburg3_sitting_static_pg_loop",
                }
                alt = tum_map.get(seq, None)
                if alt is not None:
                    pg = pgloop_dir / alt
            if (not pg.exists()) and dataset == "ICL":
                # accept a single aggregated run dir with name 'icl_all_*'
                alt = "icl_all_" + seq
                if (pgloop_dir / alt).exists():
                    pg = pgloop_dir / alt

            ape_pg = rpe_pg = None
            ape_pg_rot = rpe_pg_rot = None
            nloops = None
            loop_ms = None
            if pg.exists():
                ape_pg_p = find_any(pg, ["evo_ape_pg.zip"])
                rpe_pg_p = find_any(pg, ["evo_rpe_pg.zip"])
                ape_pg_rot_p = find_any(pg, ["evo_ape_pg_rot.zip"])
                rpe_pg_rot_p = find_any(pg, ["evo_rpe_pg_rot.zip"])

                ape_pg = rmse_from_zip(ape_pg_p) if ape_pg_p else None
                rpe_pg = rmse_from_zip(rpe_pg_p) if rpe_pg_p else None
                ape_pg_rot = rmse_from_zip(ape_pg_rot_p) if ape_pg_rot_p else None
                rpe_pg_rot = rmse_from_zip(rpe_pg_rot_p) if rpe_pg_rot_p else None
                # also accept raw zips inside pg dir if you used that format
                if ape_raw is None:
                    ape_raw_p2 = find_any(pg, ["evo_ape_raw.zip", "evo_ape.zip"])
                    ape_raw = rmse_from_zip(ape_raw_p2) if ape_raw_p2 else None
                if rpe_raw is None:
                    rpe_raw_p2 = find_any(pg, ["evo_rpe_raw.zip", "evo_rpe.zip"])
                    rpe_raw = rmse_from_zip(rpe_raw_p2) if rpe_raw_p2 else None
                if ape_raw_rot is None:
                    ape_raw_rot_p2 = find_any(pg, ["evo_ape_raw_rot.zip", "evo_ape_rot.zip"])
                    ape_raw_rot = rmse_from_zip(ape_raw_rot_p2) if ape_raw_rot_p2 else None
                if rpe_raw_rot is None:
                    rpe_raw_rot_p2 = find_any(pg, ["evo_rpe_raw_rot.zip", "evo_rpe_rot.zip"])
                    rpe_raw_rot = rmse_from_zip(rpe_raw_rot_p2) if rpe_raw_rot_p2 else None

                timing = pg / "timing_pg.json"
                if timing.exists():
                    try:
                        t = json.loads(timing.read_text(encoding="utf-8"))
                        nloops = t.get("num_loops", None)
                        loop_ms = (t.get("loop_ms", {}) or {}).get("mean", None)
                    except Exception:
                        pass

            rows.append(
                {
                    "dataset": dataset,
                    "seq": seq,
                    "ate_raw": ape_raw,
                    "rpe_raw": rpe_raw,
                    "ate_raw_rot": ape_raw_rot,
                    "rpe_raw_rot": rpe_raw_rot,
                    "ate_pg": ape_pg,
                    "rpe_pg": rpe_pg,
                    "ate_pg_rot": ape_pg_rot,
                    "rpe_pg_rot": rpe_pg_rot,
                    "num_loops": nloops,
                    "loop_mean_ms": loop_ms,
                }
            )

    add_dataset("TUM", args.tum_baseline, args.tum_pg_loop)
    add_dataset("ICL", args.icl_baseline, args.icl_pg_loop)

    # write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "seq",
                "ATE_raw_rmse",
                "RPE_raw_rmse",
                "ATE_raw_rot_rmse",
                "RPE_raw_rot_rmse",
                "ATE_loop_pgo_rmse",
                "RPE_loop_pgo_rmse",
                "ATE_loop_pgo_rot_rmse",
                "RPE_loop_pgo_rot_rmse",
                "num_loops",
                "loop_mean_ms",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["dataset"],
                    r["seq"],
                    "" if r["ate_raw"] is None else f"{r['ate_raw']:.6f}",
                    "" if r["rpe_raw"] is None else f"{r['rpe_raw']:.6f}",
                    "" if r["ate_raw_rot"] is None else f"{r['ate_raw_rot']:.6f}",
                    "" if r["rpe_raw_rot"] is None else f"{r['rpe_raw_rot']:.6f}",
                    "" if r["ate_pg"] is None else f"{r['ate_pg']:.6f}",
                    "" if r["rpe_pg"] is None else f"{r['rpe_pg']:.6f}",
                    "" if r["ate_pg_rot"] is None else f"{r['ate_pg_rot']:.6f}",
                    "" if r["rpe_pg_rot"] is None else f"{r['rpe_pg_rot']:.6f}",
                    "" if r["num_loops"] is None else str(r["num_loops"]),
                    "" if r["loop_mean_ms"] is None else f"{float(r['loop_mean_ms']):.2f}",
                ]
            )

    # write markdown
    def f4(x: Optional[float]) -> str:
        return "" if x is None else f"{x:.4f}"

    lines = []
    lines.append("# Summary: TUM & ICL (with/without Loop+PGO)\n")
    lines.append(
         "| Dataset | Sequence | ATE(raw,m) | RPE(raw,m) | APErot(raw) | RPErot(raw,deg) | ATE(loop+PGO,m) | RPE(loop+PGO,m) | APErot(loop+PGO) | RPErot(loop+PGO,deg) | #loops | loop mean ms |\n"
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
     )

    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['seq']} | {f4(r['ate_raw'])} | {f4(r['rpe_raw'])} | {f4(r.get('ate_raw_rot'))} | {f4(r.get('rpe_raw_rot'))} | {f4(r['ate_pg'])} | {f4(r['rpe_pg'])} | {f4(r.get('ate_pg_rot'))} | {f4(r.get('rpe_pg_rot'))} | {'' if r['num_loops'] is None else r['num_loops']} | {'' if r['loop_mean_ms'] is None else int(float(r['loop_mean_ms']))} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("wrote", args.out_md)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
