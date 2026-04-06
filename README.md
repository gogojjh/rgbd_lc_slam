# rgbd_lc_slam（实验性）

> **实验性项目声明**：这是一个实验性仓库，用于快速搭建并验证一个基础的 **RGB-D SLAM + 回环 + 位姿图优化（PGO）** 系统。
>
> 项目最初以 **Nanobot + 飞书** 的“vibe coding”方式迭代完成：边跑实验、边修问题、边把可复现脚本与可视化补齐，目标是尽快形成一个可对照的传统方法基线。

## 1. 功能概览

- **前端 Tracking/Odometry**：Open3D RGB-D 配准（map-to-frame）。
- **回环检索**：NetVLAD 全局描述子检索候选回环。
- **回环几何验证**：SuperPoint + LightGlue 特征匹配 + 几何一致性过滤（可选 ICP refine）。
- **后端优化**：GTSAM iSAM2 增量位姿图优化（将回环约束加入图并优化）。
- **评估**：使用 `evo` 产出 ATE/RPE（含 `--align --correct_scale`）及对应图表。

## 2. 仓库结构

- `src/rgbd_lc_slam/`：核心库代码（frontend/loop_closure/backend/io/eval/harness/cli）。
  - `frontend/`：RGB-D tracking、关键帧策略、子图构建等。
  - `loop_closure/`：检索（NetVLAD）+ 匹配（SuperPoint/LightGlue）+ 位姿估计/ICP refine。
  - `backend/`：GTSAM iSAM2 pose graph 后端。
  - `io/`：数据读取与轨迹 I/O（TUM/ICL-NUIM 兼容读取，TUM trajectory 写出）。
  - `harness/`：实验运行入口（目前仍是主要入口；负责串起 frontend/loop/backend 并落盘结果）。
  - `cli/`：命令行入口（薄封装，转发到 harness，便于以后稳定 CLI 兼容性）。
  - `eval/`：轻量评估工具（不替代 `evo`）。
- `scripts/`：可复现实验脚本
  - `scripts/datasets/`：下载与数据准备
  - `scripts/runs/`：批量运行 baseline / loop+PGO
  - `scripts/eval/`：evo 指标收集与结果汇总
  - `scripts/viz/`：结果可视化与图片打包
  - `scripts/pipeline/`：一键流水线入口（串起 run→eval→summary）
- `.github/workflows/pytest.yml`：最小 pytest CI（仅跑轻量单元测试）。
- `tests/`：单元测试（避免重依赖；open3d 不可用时会 skip）。
- `changelog/`：版本化变更记录（中文）。
- `docs/`：构建过程与实现记录（压缩版）。

> 说明：`data/`、`results/`、`outputs/`、`logs/` 默认为**本地运行产物**，已在 `.gitignore` 中忽略（避免仓库膨胀）。

## 3. 环境安装

推荐使用 conda：

```bash
cd rgbd_lc_slam
conda env create -f environment.yml
conda activate rgbd_lc_slam
```

依赖包含：Open3D、evo、GTSAM、PyTorch、LightGlue 等。

## 4. 快速上手（单序列）

```bash
# baseline（推荐用 cli 入口；目前内部仍转发到 harness）
python -m rgbd_lc_slam.cli.run_sequence --help

# loop+PGO
python -m rgbd_lc_slam.cli.run_sequence_pg --help
```

## 5. 批量跑通 Stage-1（推荐）

### 5.1 下载/准备数据

- TUM RGB-D：

```bash
python scripts/datasets/download_tum.py --help
```

- ICL-NUIM：当前仓库默认按本地路径读取（可自行准备到 `data/icl_nuim/`）。

### 5.2 一键流水线（run → eval → summary）

```bash
# 可通过环境变量指定数据根目录
export TUM_ROOT=data/tum_rgbd
export ICL_ROOT=data/icl_nuim

bash scripts/pipeline/run_stage1_all.sh
```

输出：
- 详细运行输出会放在 `results/runs/...`（本地目录）。
- 汇总表默认生成：
  - `results/summary_tum_icl_loop_pgo.md`
  - `results/summary_tum_icl_loop_pgo.csv`

## 6. 结果与可视化

- 汇总：`scripts/eval/summarize_all_results.py`
- ATE 对比图：`scripts/viz/plot_baseline_vs_pgloop.py`
- 轨迹对齐图收集：`scripts/viz/collect_evo_align_plots.py`

## 7. 版本与阶段成果

- Stage-1（传统方法基线）见：
  - `changelog/stage1_v1.0.0_rgbd_slam_loop_pgo.md`
  - `docs/BUILD_LOG_COMPRESSED.md`

## 8. 已知问题（下一阶段重点）

### 8.1 结果层面

- 部分序列 ATE/RPE 偏大；并存在“loop+PGO 反而退化”的失败案例（典型原因是错误/弱约束回环导致的图优化崩坏）。

### 8.2 工程/依赖层面

- **CI 仅安装 minimal deps**：GitHub Actions 的 pytest workflow 默认 `pip install -e .` + `pytest/numpy/scipy`，不会安装 Open3D / PyTorch / LightGlue / GTSAM，因此：
  - 涉及 heavy deps 的测试需要 `pytest.skip`（目前 frontend smoke test 会在无 open3d 时跳过）。
- **FAISS 可选依赖**：回环检索优先用 FAISS；若环境无 faiss，会自动 fallback 到 numpy brute-force（会更慢，但功能可用）。
- **数据集格式差异**：
  - ICL-NUIM 的 intrinsics 约定中 `fy` 为负；当前实现通过 `flip_y` + `fy>0` 的方式兼容（见 `io/rgbd_io.py`）。如果你自行准备的数据不符合这一假设，可能出现上下颠倒/位姿发散。

### 8.3 当前明显的结构问题

- **harness 仍承担“编排层”职责**：尽管 frontend/loop/backend/io 已基本模块化，但 `harness/run_sequence*.py` 仍负责把各模块串起来、落盘与计时；后续如果要做更严谨的 library/CLI 边界，需要进一步抽象“pipeline runner”。
- **run_sequence 与 run_sequence_pg 仍有重复**：例如数据读取、seed/track、时间统计等；后续可以合并为一个 runner + 可插拔的 backend/loop 选项。

### 一句话评价

这是一个“算法链路已成型、工程抽象逐步收敛”的结构：frontend/loop/backend/io 已成模块，但 runner 仍偏脚本化；下一步最值钱的是把 harness 的编排逻辑抽成可复用 pipeline，并把参数/配置进一步结构化。
