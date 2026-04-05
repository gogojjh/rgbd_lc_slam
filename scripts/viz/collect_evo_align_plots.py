from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd


def copy_if_exists(src: Path, dst: Path, missing: list[str]) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        missing.append(str(src))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = root / "results" / "summary_tum_icl_loop_pgo.csv"

    df = pd.read_csv(summary)

    tum_base = root / "results" / "runs" / "tum_splits_baseline"
    icl_base = root / "results" / "runs" / "icl_baseline_evo"
    tum_pg = root / "results" / "runs" / "tum_pg_loop_gpu_20260405T044719Z"
    icl_pg = root / "results" / "runs" / "icl_pg_loop_20260405T103310Z"

    tum_map = {
        "fr1_desk_full": "rgbd_dataset_freiburg1_desk_pg_loop",
        "fr1_room_full": "rgbd_dataset_freiburg1_room_pg_loop",
        "fr1_xyz_full": "rgbd_dataset_freiburg1_xyz_pg_loop",
        "fr2_desk_full": "rgbd_dataset_freiburg2_desk_pg_loop",
        "fr2_xyz_full": "rgbd_dataset_freiburg2_xyz_pg_loop",
        "fr3_long_office_household_full": "rgbd_dataset_freiburg3_long_office_household_pg_loop",
        "fr3_sitting_static_full": "rgbd_dataset_freiburg3_sitting_static_pg_loop",
    }

    out_dir = root / "results" / "trajectory_align_plots"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    missing: list[str] = []

    for _, r in df.iterrows():
        dataset = str(r["dataset"])
        seq = str(r["seq"])

        if dataset == "TUM":
            # baseline
            base_dir = tum_base / seq
            copy_if_exists(base_dir / "ate_plot_map.png", out_dir / dataset / seq / "baseline" / "ate_map.png", missing)

            # pg+loop
            pg_seq = tum_map.get(seq)
            if pg_seq:
                pg_dir = tum_pg / pg_seq
                copy_if_exists(pg_dir / "ate_plot_raw_map.png", out_dir / dataset / seq / "pgloop" / "raw_ate_map.png", missing)
                copy_if_exists(pg_dir / "ate_plot_pg_map.png", out_dir / dataset / seq / "pgloop" / "pgo_ate_map.png", missing)

        elif dataset == "ICL":
            # baseline
            base_dir = icl_base / seq
            copy_if_exists(base_dir / "ate_plot_raw_map.png", out_dir / dataset / seq / "baseline" / "raw_ate_map.png", missing)

            # pg+loop
            pg_dir = icl_pg / seq
            copy_if_exists(pg_dir / "ate_plot_raw_map.png", out_dir / dataset / seq / "pgloop" / "raw_ate_map.png", missing)
            copy_if_exists(pg_dir / "ate_plot_pg_map.png", out_dir / dataset / seq / "pgloop" / "pgo_ate_map.png", missing)
        else:
            missing.append(f"Unknown dataset row: {dataset}/{seq}")

    (out_dir / "MISSING.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")

    zip_path = root / "results" / "evo_aligned_trajectory_plots_TUM_ICL.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_dir))

    print("wrote", zip_path)
    print("missing", len(missing))


if __name__ == "__main__":
    main()
