import numpy as np


def test_frontend_seed_and_track_smoke():
    import open3d as o3d

    from rgbd_lc_slam.frontend.rgbd_icp_frontend import RGBDICPFrontend
    from rgbd_lc_slam.frontend.types import RGBDFrame, RGBDTrackingConfig

    intr = o3d.camera.PinholeCameraIntrinsic(64, 48, 50.0, 50.0, 32.0, 24.0)
    cfg = RGBDTrackingConfig(voxel=0.05, submap_k=2, icp_max_iter=10)
    fe = RGBDICPFrontend(intrinsic=intr, cfg=cfg)

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = np.ones((48, 64), dtype=np.float32) * 1.0

    r0 = fe.seed(RGBDFrame(fid=0, stamp=0.0, rgb=rgb, depth_m=depth))
    assert r0.Twc.shape == (4, 4)
    assert r0.is_keyframe

    r1 = fe.track(RGBDFrame(fid=1, stamp=0.1, rgb=rgb, depth_m=depth))
    assert r1.Twc.shape == (4, 4)
    assert isinstance(r1.is_keyframe, bool)
