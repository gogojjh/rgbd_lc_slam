import numpy as np

from rgbd_lc_slam.frontend.keyframe import should_add_keyframe


def _rotz(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = float(np.cos(th)), float(np.sin(th))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return R


def test_should_add_keyframe_translation_threshold():
    Twc0 = np.eye(4)
    Twc1 = np.eye(4)
    Twc1[:3, 3] = [0.2, 0.0, 0.0]

    assert should_add_keyframe(Twc0, Twc1, keyframe_trans=0.1, keyframe_rot_deg=10.0)
    assert not should_add_keyframe(Twc0, Twc1, keyframe_trans=0.3, keyframe_rot_deg=10.0)


def test_should_add_keyframe_rotation_threshold():
    Twc0 = np.eye(4)
    Twc1 = np.eye(4)
    Twc1[:3, :3] = _rotz(15.0)

    assert should_add_keyframe(Twc0, Twc1, keyframe_trans=1.0, keyframe_rot_deg=10.0)
    assert not should_add_keyframe(Twc0, Twc1, keyframe_trans=1.0, keyframe_rot_deg=20.0)
