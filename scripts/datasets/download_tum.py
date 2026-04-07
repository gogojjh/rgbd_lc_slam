#!/usr/bin/env python3
"""Download and extract selected TUM RGB-D sequences.

Usage:
  python scripts/datasets/download_tum.py --out data/tum_rgbd --seq fr1/desk fr2/xyz

Notes:
- Uses official TUM RGB-D dataset URLs (vision.in.tum.de), which redirect to the
  current cvg.cit.tum.de mirror.
- Uses Python stdlib (urllib) to download.
- Supports resumable downloads and time-bounded runs to cope with cluster/CI
  timeouts.
"""

from __future__ import annotations

import argparse
import tarfile
import time
import urllib.error
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


def _open(url: str, *, start: int) -> urllib.response.addinfourl:
    headers = {}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=60)


def download(
    url: str,
    dst: Path,
    *,
    chunk_mb: int = 16,
    max_seconds: float | None = None,
) -> bool:
    """Download url -> dst.

    Returns True if the file is fully downloaded, False if it stopped early due
    to max_seconds.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    chunk_bytes = int(chunk_mb) * 1024 * 1024

    have = dst.stat().st_size if dst.exists() else 0
    print(f"+ download {url} -> {dst} (have={have})", flush=True)

    while True:
        if max_seconds is not None and (time.monotonic() - t0) >= float(max_seconds):
            print(f"  stop early (max_seconds={max_seconds}), have={have}")
            return False

        try:
            with _open(url, start=have) as r:
                status = getattr(r, "status", None)
                total_s = r.getheader("Content-Length")

                # If we asked for a range but the server ignored it, restart.
                if have > 0 and status == 200:
                    print("  server ignored Range; restarting", flush=True)
                    dst.unlink(missing_ok=True)
                    have = 0

                mode = "ab" if have > 0 else "wb"
                with dst.open(mode) as f:
                    while True:
                        if max_seconds is not None and (time.monotonic() - t0) >= float(max_seconds):
                            have = dst.stat().st_size
                            print(f"  stop early (max_seconds={max_seconds}), have={have}")
                            return False

                        chunk = r.read(chunk_bytes)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)

                        if total_s is not None:
                            total = int(total_s)
                            pct = 100.0 * (have / total) if total > 0 else 0.0
                            print(f"  {have}/{total} ({pct:.1f}%)", flush=True)
                        else:
                            print(f"  {have} bytes", flush=True)

            # If the server provides Content-Length, we can decide completion.
            if total_s is not None and have >= int(total_s):
                return True

            # If not, assume done once the stream ends.
            if total_s is None:
                return True

            # Otherwise, loop and request the next range.

        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP error downloading {url}: {e}") from e


def extract_tgz(tgz_path: Path, out_dir: Path) -> None:
    print(f"+ extract {tgz_path} -> {out_dir}", flush=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/tum_rgbd"))
    ap.add_argument("--seq", nargs="+", required=True, help="e.g., fr1/desk fr2/xyz")
    ap.add_argument("--no_extract", action="store_true")
    ap.add_argument("--chunk_mb", type=int, default=16)
    ap.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="stop download early after N seconds (resume next run)",
    )
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    for seq in args.seq:
        url, tgz = url_for(seq)
        tgz_path = out / tgz
        seq_dir = out / tgz.replace(".tgz", "")

        done = download(url, tgz_path, chunk_mb=args.chunk_mb, max_seconds=args.max_seconds)
        if not done:
            continue

        if not args.no_extract:
            # TUM tgz already contains a folder named like rgbd_dataset_freiburgX_...
            if seq_dir.exists() and any(seq_dir.iterdir()):
                print(f"skip extract (exists): {seq_dir}")
            else:
                extract_tgz(tgz_path, out)

    print("done")


if __name__ == "__main__":
    main()
