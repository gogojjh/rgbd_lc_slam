# rgbd_lc_slam

RGB-D SLAM with Open3D frontend (map-to-frame registration), NetVLAD loop closure retrieval, SuperPoint+LightGlue geometric verification, and GTSAM pose-graph optimization (iSAM2).

## Quick start

```bash
cd /Titan/code/robohike_ws/src/rgbd_lc_slam
conda env create -f environment.yml
conda activate rgbd_lc_slam

# (1) download TUM RGB-D sequences
python scripts/download_tum.py --help

# (2) run one sequence
python -m rgbd_lc_slam.harness.run_sequence --help

# (3) evaluate with evo
python -m rgbd_lc_slam.eval.evo_eval --help
```

## Repo layout
- `src/rgbd_lc_slam/`: library code
- `scripts/`: data download/setup scripts
- `configs/`: dataset splits + run configs
- `results/`: generated metrics/plots/trajectories

## Notes
- Frontend latency target: <30ms/frame for registration (loop closure + backend run asynchronously).
- Evaluation: ATE/RPE using `evo`.
