#!/usr/bin/env python3
"""Download and extract selected TUM RGB-D sequences.

Usage:
  python scripts/download_tum.py --out data/tum_rgbd --seq fr1/desk fr2/xyz

Notes:
- Uses official TUM RGB-D dataset URLs.
- Uses Python stdlib (urllib) to download, to avoid external wget dependency.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://vision.in.tum.de/rgbd/dataset/freiburg"


def url_for(seq: str) -> tuple[str, str]:
    # seq like 'fr1/desk' => 'rgbd_dataset_freiburg1_desk.tgz'
    seq = seq.strip().replace("\\", "/")
    if not seq.startswith("fr"):
        raise ValueError(f"Expected seq like fr1/desk, got {seq}")
    fr, name = seq.split("/", 1)
    fr_num = fr.replace("fr", "")
    tgz = f"rgbd_dataset_freiburg{fr_num}_{name.replace('/', '_')}.tgz"
    return f"{BASE}{fr_num}/{tgz}", tgz


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ download {url} -> {dst}", flush=True)
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def extract_tgz(tgz_path: Path, out_dir: Path) -> None:
    print(f"+ extract {tgz_path} -> {out_dir}", flush=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/tum_rgbd"))
    ap.add_argument("--seq", nargs="+", required=True, help="e.g., fr1/desk fr2/xyz")
    ap.add_argument("--no_extract", action="store_true")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    for seq in args.seq:
        url, tgz = url_for(seq)
        tgz_path = out / tgz
        seq_dir = out / tgz.replace(".tgz", "")

        if not tgz_path.exists():
            download(url, tgz_path)
        else:
            print(f"skip download: {tgz_path}")

        if not args.no_extract:
            # TUM tgz already contains a folder named like rgbd_dataset_freiburgX_...
            if seq_dir.exists() and any(seq_dir.iterdir()):
                print(f"skip extract (exists): {seq_dir}")
            else:
                extract_tgz(tgz_path, out)

    print("done")


if __name__ == "__main__":
    main()
