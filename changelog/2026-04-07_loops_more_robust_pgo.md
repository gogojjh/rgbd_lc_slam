# 版本：unreleased - 日期：2026-04-07

## 变更概述

- 目标：提高回环触发数量（`num_loops`），并在 PGO 侧引入 robust kernel（Huber/Cauchy）提升对错误回环(outlier)的鲁棒性。
- 背景：在 TUM `fr1_xyz` 的最小验证（前 200 帧）中，回环数量较少且存在“加入错误回环后 PGO 退化”的风险，需要同时从“召回更多候选回环”和“后端鲁棒化”两侧推进。

## 核心变更

### Loop closure（提高 num_loops 的默认倾向）

- 调整 `LoopClosureConfig` 默认值（更偏向召回更多候选并允许更多几何验证尝试）：
  - `retrieval_top_k: 10 -> 20`
  - `exclude_recent: 30 -> 15`
  - `max_verify_per_frame: 3 -> 10`
  - 保持 `max_rmse_m=0.02`（用于抑制明显的几何误差）
  - 保持 `min_inliers=100`（更强的几何一致性下限）

### Backend / PGO（robust kernel）

- `ISAM2BackendConfig` 已支持对回环 between factors 启用 robust kernel：
  - `robust_loop_factors`（开关）
  - `robust_loop_kernel`（`huber`/`cauchy`）
  - `robust_loop_param`（核函数参数 k）
- 将这些配置暴露为 `run_sequence_pg` 的 CLI 参数，便于快速实验与参数 sweep。

### CLI（参数暴露与默认更友好）

- `python -m rgbd_lc_slam.cli.run_sequence_pg` 增加并转发参数：
  - `--retrieval_min_score_margin`
  - `--min_inlier_ratio`
  - `--loop_every_kf`
  - `--robust_loop_factors / --no-robust_loop_factors`
  - `--robust_loop_kernel`
  - `--robust_loop_param`

## 简单测试（TUM fr1_xyz, 200 帧）

命令（Huber）：

```bash
python -m rgbd_lc_slam.harness.run_sequence_pg \
  --seq_dir data/tum_rgbd/rgbd_dataset_freiburg1_xyz \
  --out_dir results/runs/phase3_loops_more_huber_tum_fr1_xyz_200_20260407_095744 \
  --max_frames 200 --enable_loop --device cpu \
  --exclude_recent 15 --retrieval_top_k 20 --retrieval_min_score 0.7 \
  --min_inlier_ratio 0.35 --loop_every_kf 2 \
  --robust_loop_factors --robust_loop_kernel huber --robust_loop_param 1.0
```

- `num_loops`: 3（相比此前 1 有明显提升）
- ATE RMSE（align+correct_scale）：raw 0.057518 -> pg 0.048962

对照（禁用 robust kernel，其他一致）：

```bash
python -m rgbd_lc_slam.harness.run_sequence_pg \
  --seq_dir data/tum_rgbd/rgbd_dataset_freiburg1_xyz \
  --out_dir results/runs/phase3_loops_more_norobust_tum_fr1_xyz_200_20260407_095907 \
  --max_frames 200 --enable_loop --device cpu \
  --exclude_recent 15 --retrieval_top_k 20 --retrieval_min_score 0.7 \
  --min_inlier_ratio 0.35 --loop_every_kf 2 \
  --no-robust_loop_factors
```

- `num_loops`: 3
- ATE RMSE（align+correct_scale）：pg 0.049050

备注：该最小用例下，Huber 与 no-robust 的 ATE 很接近；robust kernel 的价值更可能体现在“存在明显 outlier 回环”的序列或更长序列中。

## 后续行动（未来改进）

- 回环质量控制（比单纯阈值更有效）：
  - 增加基于几何退化的拒绝/降权（例如平面退化、低视差、深度有效像素比例）。
  - 引入 switchable constraints / dynamic covariance scaling（DCS）等更强的 outlier 抑制方法。
  - 引入回环候选多假设（多候选进入后端，由后端鲁棒机制自动选择）。
- 参数 sweep：提供脚本在多个 TUM/ICL 序列上批量 sweep `exclude_recent/top_k/min_score/min_inlier_ratio/robust_param`，输出 “num_loops vs ATE” 的表格。
- 性能：`max_verify_per_frame` 增大后 loop_ms p99 变大；后续可加入超时、并行或更快的 early-stop 策略。
