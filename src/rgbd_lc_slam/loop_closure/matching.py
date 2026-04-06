from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def _preload_conda_runtime_libs() -> None:
    """Best-effort preload conda runtime libs (libstdc++/libgcc) as RTLD_GLOBAL.

    This helps when other native modules accidentally load an older system
    libstdc++ first, which can later break importing cv2/lightglue.

    Note: preloading is not always sufficient if the old system libstdc++ is
    already loaded, so we *also* enforce importing cv2 before importing torch.
    """

    prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    libdir = Path(prefix) / "lib"
    for name in ("libgcc_s.so.1", "libstdc++.so.6"):
        p = libdir / name
        if not p.exists():
            continue
        try:
            ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass


def _best_effort_import_cv2() -> Optional[Any]:
    """Import cv2 in the safest possible way.

    Key rule for this repo: **cv2 must be imported before torch/lightglue**,
    otherwise torch may load a system libstdc++ that is too old for cv2.

    Returns cv2 module or None.
    """

    _preload_conda_runtime_libs()
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _has_superpoint_weights() -> bool:
    """Return True if SuperPoint weights seem present in torch hub cache.

    Motivation: in some environments, LightGlue will try to download weights
    on first import/instantiation, which can block long runs (no internet / slow).

    If weights are missing, we skip LightGlue and fall back to ORB immediately.
    """

    torch_home = os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
    ckdir = Path(torch_home) / "hub" / "checkpoints"
    if not ckdir.exists():
        return False

    # Any non-partial file starting with superpoint_v1.pth is treated as available.
    for p in ckdir.glob("superpoint_v1.pth*"):
        if p.name.endswith(".partial"):
            continue
        try:
            if p.stat().st_size > 5 * 1024 * 1024:  # heuristic: >5MB
                return True
        except Exception:
            continue
    return False


def _to_gray_np(image: np.ndarray) -> np.ndarray:
    """Convert RGB or grayscale numpy image to grayscale float32 in [0,1]."""

    if image.ndim == 3 and image.shape[2] == 3:
        gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(
            np.float32
        )
    elif image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        raise ValueError("image must be HxW or HxWx3")

    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


@dataclass
class MatchResult:
    kpts0: np.ndarray  # (N,2)
    kpts1: np.ndarray  # (N,2)
    scores: np.ndarray  # (N,)


class SuperPointLightGlueMatcher:
    """SuperPoint + LightGlue, with an ABI-safe import order and robust fallbacks.

    Backends (best-effort):
      1) LightGlue + SuperPoint (requires cv2 + torch + weights)
      2) OpenCV ORB + BFMatcher (requires cv2)
      3) Dummy matcher returning 0 matches (when cv2 cannot be imported)

    Design goal: long runs should never crash/hang due to cv2/lightglue ABI or
    on-the-fly weight downloads.
    """

    def __init__(
        self,
        device: Union[str, object] = "cuda",
        max_num_keypoints: int = 2048,
    ):
        self.max_num_keypoints = int(max_num_keypoints)

        # 1) Import cv2 first (critical for ABI stability). If cv2 import fails due
        # to C++ ABI issues (common on mixed system/conda setups), we still keep the
        # system running by falling back to a dummy matcher.
        self._cv2 = _best_effort_import_cv2()

        # Internal state
        self._mode: str
        self._device = device
        self._torch = None
        self.extractor = None
        self.matcher = None
        self._orb = None
        self._bf = None

        # 2) Try LightGlue (imports torch)
        try:
            if self._cv2 is None:
                raise ImportError("cv2 not available")

            # Avoid blocking downloads in long runs: if weights are missing, use ORB.
            if not _has_superpoint_weights():
                raise ImportError("SuperPoint weights not found in cache; skip LightGlue")

            _preload_conda_runtime_libs()
            import torch  # type: ignore

            from lightglue import LightGlue, SuperPoint  # type: ignore

            self._torch = torch
            self.device = torch.device(device)  # type: ignore

            self.extractor = (
                SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
            )
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            self._mode = "lightglue"
            return
        except Exception:
            pass

        # 3) Fallback to ORB
        if self._cv2 is not None:
            cv2 = self._cv2
            self._orb = cv2.ORB_create(nfeatures=self.max_num_keypoints)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self._mode = "orb"
            self.device = None
            return

        # 4) Last resort: dummy matcher (no matches)
        self._mode = "none"
        self.device = None

    def _to_gray_tensor(self, image: np.ndarray):
        assert self._torch is not None
        torch = self._torch
        gray = _to_gray_np(image)
        return torch.from_numpy(gray)[None, None].to(self.device)  # (1,1,H,W)

    def match(self, image0: np.ndarray, image1: np.ndarray) -> MatchResult:
        if self._mode == "none":
            return MatchResult(
                kpts0=np.zeros((0, 2), dtype=np.float32),
                kpts1=np.zeros((0, 2), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )

        if self._mode == "lightglue":
            assert self._torch is not None
            torch = self._torch
            assert self.extractor is not None and self.matcher is not None

            with torch.no_grad():
                t0 = self._to_gray_tensor(image0)
                t1 = self._to_gray_tensor(image1)

                feats0 = self.extractor.extract(t0)
                feats1 = self.extractor.extract(t1)

                matches = self.matcher({"image0": feats0, "image1": feats1})
                m0 = matches["matches0"][0].detach().cpu().numpy()  # (K0,), -1 if no match
                s0 = matches.get("matching_scores0", None)
                if s0 is not None:
                    s0 = s0[0].detach().cpu().numpy()
                else:
                    s0 = np.ones_like(m0, dtype=np.float32)

                kpts0 = feats0["keypoints"][0].detach().cpu().numpy()  # (K0,2)
                kpts1 = feats1["keypoints"][0].detach().cpu().numpy()  # (K1,2)

                valid = m0 >= 0
                idx1 = m0[valid].astype(np.int64)
                out0 = kpts0[valid]
                out1 = kpts1[idx1]
                scores = s0[valid].astype(np.float32)

                return MatchResult(kpts0=out0, kpts1=out1, scores=scores)

        # ORB fallback
        assert self._cv2 is not None and self._orb is not None and self._bf is not None
        cv2 = self._cv2

        g0 = _to_gray_np(image0)
        g1 = _to_gray_np(image1)
        g0u = (np.clip(g0, 0.0, 1.0) * 255.0).astype(np.uint8)
        g1u = (np.clip(g1, 0.0, 1.0) * 255.0).astype(np.uint8)

        k0, d0 = self._orb.detectAndCompute(g0u, None)
        k1, d1 = self._orb.detectAndCompute(g1u, None)
        if d0 is None or d1 is None or k0 is None or k1 is None or len(k0) == 0 or len(k1) == 0:
            return MatchResult(
                kpts0=np.zeros((0, 2), dtype=np.float32),
                kpts1=np.zeros((0, 2), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )

        ms = self._bf.match(d0, d1)
        if len(ms) == 0:
            return MatchResult(
                kpts0=np.zeros((0, 2), dtype=np.float32),
                kpts1=np.zeros((0, 2), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )

        ms = sorted(ms, key=lambda m: m.distance)[: self.max_num_keypoints]
        pts0 = np.array([k0[m.queryIdx].pt for m in ms], dtype=np.float32)
        pts1 = np.array([k1[m.trainIdx].pt for m in ms], dtype=np.float32)
        dist = np.array([m.distance for m in ms], dtype=np.float32)
        scores = 1.0 / (1.0 + dist)

        return MatchResult(kpts0=pts0, kpts1=pts1, scores=scores)
