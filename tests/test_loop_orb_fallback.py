import os
import numpy as np


def test_matcher_fallback_without_weights(monkeypatch):
    # Force matcher to believe torch hub cache has no superpoint weights
    monkeypatch.setenv('TORCH_HOME', '/tmp/torch_home_empty_for_test')

    from rgbd_lc_slam.loop_closure.matching import SuperPointLightGlueMatcher

    m = SuperPointLightGlueMatcher(device='cpu', max_num_keypoints=200)
    assert getattr(m, '_mode', None) in ('orb', 'none')

    img0 = (np.random.rand(60, 80, 3) * 255).astype(np.uint8)
    img1 = img0.copy()
    res = m.match(img0, img1)
    assert res.kpts0.shape[1] == 2
    assert res.kpts1.shape[1] == 2
    assert res.scores.ndim == 1
