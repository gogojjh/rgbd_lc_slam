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

### Scripts（可复现跑测与下载增强）

- `scripts/datasets/download_tum.py` 支持断点续传与 `--max_seconds` 限时下载（适配集群/CI 中断），并提供下载进度输出。
- `scripts/runs/run_all_tum_splits.sh`、`scripts/runs/run_all_tum_splits_pg_loop.sh`：统一走 `conda run -n rgbd_lc_slam`，并设置 `MPLBACKEND=Agg` 以适配 headless 环境。
- `scripts/runs/run_all_icl_nuim_pg_loop.sh`：支持通过入参指定 `ICL_ROOT/OUT_RUNS_DIR`，并修正 `evo_rpe` 旋转误差参数到 `-r angle_deg`。
- 新增多组辅助脚本（最小验证/论文对比）：
  - `scripts/runs/run_one_tum_pg_se3_highres.sh`（单序列 SE3 + voxel=0.02）
  - `scripts/runs/run_tum_fr1_mast3r_subset_no_download.sh`、`scripts/runs/run_tum_fr1_mast3r_subset_pg_se3_highres.sh`
  - `scripts/eval/extract_mast3r_slam_tum_table.py`、`scripts/eval/plot_gt_baseline_loop_pgo_compare.py`

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

## 全量跑测（TUM+ICL 全序列，baseline vs loop+PGO）

本次全量跑测基于当前代码版本（含：提高回环召回倾向 + loop between factor robust kernel 默认开启），跑测目录：

- `results/runs/phase4_fullsuite_20260407T041034Z`
- 汇总表：`results/summary_tum_icl_loop_pgo.md`、`results/summary_tum_icl_loop_pgo.csv`

### TUM（7 序列）

结论：loop+PGO 在 7/7 序列上 ATE/RPE 均有提升（相对 baseline）。

- `fr1_desk`: ATE 0.3777 → 0.0944；RPE 0.0483 → 0.0215；loops=7
- `fr1_room`: ATE 0.2692 → 0.1357；RPE 0.0194 → 0.0094；loops=7
- `fr1_xyz`: ATE 0.0683 → 0.0274；RPE 0.0172 → 0.0081；loops=16
- `fr2_desk`: ATE 0.3837 → 0.0574；RPE 0.0152 → 0.0093；loops=28
- `fr2_xyz`: ATE 0.1224 → 0.0359；RPE 0.0119 → 0.0072；loops=20
- `fr3_long_office_household`: ATE 0.7061 → 0.0956；RPE 0.0136 → 0.0074；loops=31
- `fr3_sitting_static`: ATE 0.0855 → 0.0099；RPE 0.0080 → 0.0050；loops=0（PGO 仍改善，主要来自位姿图平滑）

### ICL-NUIM（8 序列）

结论：ICL 上 ATE 多数持平/小幅改善，但 RPE 改善不稳定（部分序列略变差）。

- 典型改善：`office_traj2`: ATE 0.0258 → 0.0196；RPE 0.0029 → 0.0029；loops=2
- 典型变差：`living_room_traj0`: ATE 0.1485 → 0.1084（改善）但 RPE 0.0093 → 0.0108（变差）

备注：ICL 的 loop 触发数量普遍较少（0~6），且 loop_ms 非常低（~1–6ms）；后续如要提升 ICL 端收益，可能需要单独调 `exclude_recent/top_k/min_score` 等参数。


- 回环质量控制（比单纯阈值更有效）：
  - 增加基于几何退化的拒绝/降权（例如平面退化、低视差、深度有效像素比例）。
  - 引入 switchable constraints / dynamic covariance scaling（DCS）等更强的 outlier 抑制方法。
  - 引入回环候选多假设（多候选进入后端，由后端鲁棒机制自动选择）。
- 参数 sweep：提供脚本在多个 TUM/ICL 序列上批量 sweep `exclude_recent/top_k/min_score/min_inlier_ratio/robust_param`，输出 “num_loops vs ATE” 的表格。
- 性能：`max_verify_per_frame` 增大后 loop_ms p99 变大；后续可加入超时、并行或更快的 early-stop 策略。
