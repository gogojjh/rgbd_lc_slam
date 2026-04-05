# Repository Guidelines

## Project Structure & Module Organization
This repository uses a `src/` layout. Core package code lives in `src/rgbd_lc_slam/`, split by responsibility: `frontend/` for frame processing, `loop_closure/` for retrieval and geometric verification, `backend/` for GTSAM pose-graph optimization, `io/` for dataset and trajectory utilities, `eval/` for metrics, and `harness/` for runnable entry points such as `run_sequence.py`. Use `scripts/` for dataset download, batch runs, and result summarization. Keep static configuration in `configs/`, input datasets under `data/`, and generated artifacts in `outputs/`, `logs/`, and `results/`.

## Build, Test, and Development Commands
Set up the environment with `conda env create -f environment.yml` and `conda activate rgbd_lc_slam`. Install the package in editable mode with `pip install -e .` so `python -m rgbd_lc_slam...` resolves local changes.

- `python scripts/download_tum.py --help`: inspect dataset download options.
- `python -m rgbd_lc_slam.harness.run_sequence --help`: run the baseline RGB-D pipeline on one sequence.
- `python -m rgbd_lc_slam.harness.run_sequence_pg --help`: run the pose-graph / loop-closure variant.
- `python -m rgbd_lc_slam.eval.evo_eval --help`: evaluate trajectories with `evo`.
- `pytest`: run Python tests once they are added.
- `ruff check src scripts` and `black src scripts`: lint and format code.

## Coding Style & Naming Conventions
Target Python 3.11, follow PEP 8, and use 4-space indentation. Prefer type hints and `pathlib.Path`, matching existing modules. Use `snake_case` for functions, variables, and file names; use `PascalCase` for classes such as `PoseGraphISAM2Backend`. Keep CLI modules small and push reusable logic into package modules under `src/rgbd_lc_slam/`.

## Testing Guidelines
There is no dedicated `tests/` tree yet; add new tests under `tests/` with names like `test_tum_reader.py`. Favor `pytest` unit tests for I/O, metrics, and pose utilities, plus small regression fixtures over large datasets. For pipeline changes, include the exact command used and the output directory, for example `outputs/icl_living_room_traj0_loop_pgo/`.

## Commit & Pull Request Guidelines
This repository currently has no commit history, so use concise imperative commit subjects such as `Add loop-closure trajectory export`. Keep commits scoped to one change. Pull requests should describe the scenario tested, list commands run, note any dataset or config dependencies, and attach representative plots or metric summaries when behavior changes.
