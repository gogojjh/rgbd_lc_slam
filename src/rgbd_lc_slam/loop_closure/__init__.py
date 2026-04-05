"""Loop closure module for rgbd_lc_slam.

Implements:
- NetVLAD global descriptor computation (pretrained weights optional)
- Retrieval database (brute force or FAISS if available)
- SuperPoint + LightGlue feature matching
- RGB-D geometric verification (depth-assisted 3D-3D Umeyama+RANSAC or PnP)
- Optional ICP refinement

This package is intentionally lightweight and designed as a code skeleton.
"""

from .types import CameraIntrinsics, LoopClosureConfig, LoopConstraint, RetrievalCandidate
from .netvlad import NetVLADDescriptor
from .retrieval_db import RetrievalDatabase
from .matching import SuperPointLightGlueMatcher
from .pose_estimation import RGBDPoseEstimator
from .icp_refine import ICPRefiner
from .loop_closure import LoopClosureModule

__all__ = [
    "CameraIntrinsics",
    "LoopClosureConfig",
    "LoopConstraint",
    "RetrievalCandidate",
    "NetVLADDescriptor",
    "RetrievalDatabase",
    "SuperPointLightGlueMatcher",
    "RGBDPoseEstimator",
    "ICPRefiner",
    "LoopClosureModule",
]
