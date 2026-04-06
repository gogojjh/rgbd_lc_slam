import numpy as np

from rgbd_lc_slam.io.tum_reader import associate_by_time


def test_associate_by_time_basic():
    rgb = [(0.00, "rgb/0.png"), (0.10, "rgb/1.png"), (0.21, "rgb/2.png")]
    depth = [(0.01, "depth/0.png"), (0.11, "depth/1.png"), (0.20, "depth/2.png")]

    pairs = associate_by_time(rgb, depth, max_dt=0.02)
    assert len(pairs) == 3
    # ensure matched in order
    assert pairs[0][1].endswith("0.png")
    assert pairs[1][1].endswith("1.png")
    assert pairs[2][1].endswith("2.png")


def test_associate_by_time_reject_far():
    rgb = [(0.00, "rgb/0.png")]
    depth = [(0.50, "depth/0.png")]

    pairs = associate_by_time(rgb, depth, max_dt=0.02)
    assert pairs == []
