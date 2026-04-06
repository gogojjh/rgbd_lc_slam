import numpy as np

from rgbd_lc_slam.io.trajectory_io import write_tum_trajectory


def test_write_tum_trajectory_format(tmp_path):
    path = tmp_path / "traj.txt"

    stamps = [0.0, 0.1]
    Twc0 = np.eye(4)
    Twc1 = np.eye(4)
    Twc1[:3, 3] = [1.0, 2.0, 3.0]

    write_tum_trajectory(path, stamps, [Twc0, Twc1])

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    for l in lines:
        parts = l.split()
        assert len(parts) == 8
        # timestamps and numbers should be parseable
        [float(x) for x in parts]
