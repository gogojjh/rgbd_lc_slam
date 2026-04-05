from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .utils import l2_normalize


class NetVLADLayer(nn.Module):
    """NetVLAD pooling layer.

    This is a standard implementation: soft-assignment to K clusters followed by
    residual aggregation and intra-normalization.

    Outputs a (B, K*D) vector.
    """

    def __init__(self, num_clusters: int = 64, dim: int = 512, normalize_input: bool = True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        B, D, H, W = x.shape
        soft_assign = self.conv(x).view(B, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(B, D, -1)

        # Compute residuals to each centroid
        # residual: (B, K, D, HW)
        residual = x_flatten.unsqueeze(1) - self.centroids.unsqueeze(-1).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)

        vlad = residual.sum(dim=-1)  # (B, K, D)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(B, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class NetVLADModel(nn.Module):
    """Backbone + NetVLAD + optional projection."""

    def __init__(
        self,
        num_clusters: int = 64,
        out_dim: int = 4096,
        backbone: str = "vgg16",
    ):
        super().__init__()
        if backbone != "vgg16":
            raise ValueError("Only vgg16 backbone skeleton is implemented")

        # VGG16 conv features
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.backbone = vgg.features
        self.pool = NetVLADLayer(num_clusters=num_clusters, dim=512)

        vlad_dim = num_clusters * 512
        self.proj = nn.Linear(vlad_dim, out_dim) if out_dim is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # (B, 512, H/32, W/32)
        vlad = self.pool(feat)
        if self.proj is not None:
            vlad = self.proj(vlad)
            vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


@dataclass
class NetVLADWeights:
    path: Optional[str] = None


class NetVLADDescriptor:
    """Computes global image descriptors using NetVLAD.

    Notes:
      - This class can load pretrained NetVLAD weights from a local checkpoint.
      - If no weights are provided/found, it falls back to a *simple but stable*
        image descriptor (64x64 grayscale flatten + L2 norm). This is not SOTA,
        but is good enough to enable retrieval + loop closure without requiring
        external checkpoints.

    Checkpoint format expectation (flexible): a dict with key "state_dict" or a
    raw model state dict.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        num_clusters: int = 64,
        out_dim: int = 4096,
        weights: Optional[NetVLADWeights] = None,
        *,
        resize_hw: Optional[tuple[int, int]] = (224, 224),
    ):
        self.device = torch.device(device)
        self.model = NetVLADModel(num_clusters=num_clusters, out_dim=out_dim).to(self.device).eval()
        self.resize_hw = resize_hw

        self._weights_loaded = False
        if weights and weights.path:
            self._weights_loaded = self._try_load(weights.path)

        # Normalization consistent with torchvision VGG pretraining
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _try_load(self, path: str) -> bool:
        if not os.path.exists(path):
            # Keep random init for NetVLAD layer + proj
            return False
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Allow checkpoints saved with prefixes
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_state[k[len("model.") :]] = v
            else:
                new_state[k] = v
        self.model.load_state_dict(new_state, strict=False)
        return True

    @staticmethod
    def _simple_fallback_desc(rgb: np.ndarray, size_hw: tuple[int, int] = (64, 64)) -> np.ndarray:
        """Very lightweight descriptor when NetVLAD weights are unavailable.

        - convert to grayscale float32 in [0,1]
        - resize to 64x64
        - flatten to 4096-D and L2 normalize
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected rgb as (H,W,3)")
        x = rgb.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

        # resize (try cv2, else numpy fallback)
        try:
            import cv2  # type: ignore

            g = cv2.resize(gray, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            # crude fallback: center crop/pad then reshape
            g = gray
            g = g[: size_hw[0], : size_hw[1]]
            if g.shape != size_hw:
                out = np.zeros(size_hw, dtype=np.float32)
                out[: g.shape[0], : g.shape[1]] = g
                g = out

        d = g.reshape(-1).astype(np.float32)
        d = l2_normalize(d)
        return d

    @torch.no_grad()
    def compute(self, rgb: np.ndarray) -> np.ndarray:
        """Compute L2-normalized descriptor for an RGB image.

        Args:
            rgb: uint8 or float image (H,W,3) in RGB order.

        Returns:
            desc: (D,) float32 L2-normalized.
        """
        if not self._weights_loaded:
            # Stable fallback to avoid random NetVLAD behavior
            return self._simple_fallback_desc(rgb, size_hw=(64, 64))

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected rgb as (H,W,3)")

        x = rgb
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0

        # (1,3,H,W)
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Speed: downsample to a fixed size before VGG16 if requested.
        if self.resize_hw is not None:
            t = F.interpolate(t, size=self.resize_hw, mode="bilinear", align_corners=False)

        t = (t - self.mean) / self.std
        desc = self.model(t).squeeze(0).float().cpu().numpy()
        desc = l2_normalize(desc.astype(np.float32))
        return desc
