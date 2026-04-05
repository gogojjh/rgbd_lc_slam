# rgbd_lc_slam（实验性）

> **实验性项目声明**：这是一个实验性仓库，用于快速搭建并验证一个基础的 **RGB-D SLAM + 回环 + 位姿图优化（PGO）** 系统。
>
> 项目最初以 **openclaw + 飞书** 的“vibe coding”方式迭代完成：边跑实验、边修问题、边把可复现脚本与可视化补齐，目标是尽快形成一个可对照的传统方法基线。

## 1. 功能概览

- **前端 Tracking/Odometry**：Open3D RGB-D 配准（map-to-frame）。
- **回环检索**：NetVLAD 全局描述子检索候选回环。
- **回环几何验证**：SuperPoint + LightGlue 特征匹配 + 几何一致性过滤（可选 ICP refine）。
- **后端优化**：GTSAM iSAM2 增量位姿图优化（将回环约束加入图并优化）。
- **评估**：使用 `evo` 产出 ATE/RPE（含 `--align --correct_scale`）及对应图表。

## 2. 仓库结构

- `src/rgbd_lc_slam/`：核心库代码（前端/回环/后端/评估）。
- `configs/`：数据集/序列配置（如 TUM splits）。
- `scripts/`：可复现实验脚本
  - `scripts/datasets/`：下载与数据准备
  - `scripts/runs/`：批量运行 baseline / loop+PGO
  - `scripts/eval/`：evo 指标收集与结果汇总
  - `scripts/viz/`：结果可视化与图片打包
  - `scripts/pipeline/`：一键流水线入口（串起 run→eval→summary）
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
# baseline
python -m rgbd_lc_slam.harness.run_sequence --help

# loop+PGO
python -m rgbd_lc_slam.harness.run_sequence_pg --help
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

- 部分序列 ATE/RPE 偏大；并存在“loop+PGO 反而退化”的失败案例。
- 下一步需要强化回环约束 gating（多阶段验证、鲁棒核、switchable constraints）与前端稳健性。

### 当前明显的结构问题

- frontend 目录基本是空的，但实际前端跟踪逻辑写在 harness/run_sequence.py 里，这说明“前端”还没有真正模块化。
- harness/run_sequence.py 和 harness/run_sequence_pg.py 存在较多重复代码，例如图像读取、点云构建、子图构建、关键帧判定，后续维护会有同步修改风险。
- 目前没有 tests/，说明结构虽清楚，但缺少回归保护。
- scripts/、outputs/、results/ 比较重，工程偏实验驱动；如果后续继续扩展，建议把实验脚本和核心库边界再拉开。

### 一句话评价

这是一个“算法链路已经成型、工程抽象还差最后一步”的结构：回环和后端模块化程度不错，但前端仍嵌在运行脚本里，下一步最值得做的是把 tracking/front-end 从 harness 中抽成真正的 frontend 模块。
