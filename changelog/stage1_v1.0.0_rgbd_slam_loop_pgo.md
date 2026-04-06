# 版本：v1.0.0 - 日期：2026-04-05 - 代码提交：664ecd0 - changelog更新提交：8b61c8f

## 变更概述

- 完成了一个可复现实验流水线的 RGB-D SLAM + 回环 + 位姿图优化（PGO）基线系统，并在 TUM RGB-D 与 ICL-NUIM 两个数据集上产出可对比的 ATE/RPE 指标与轨迹可视化结果。
- 问题现象：过去缺少一套“端到端可跑通”的传统 SLAM 基线，导致无法快速验证回环模块、PGO 以及不同参数对精度/速度的影响；评估结果分散，难以汇总对比。
- 修改目标：建立一个脚本化、可批量运行、可自动评估（evo）与可视化的 RGB-D SLAM 实验仓库，为后续算法优化与学习方法对照提供稳定参照。

## 核心变更

### 新增

- 新增完整的 RGB-D SLAM 流水线：
  - 前端：基于 Open3D 的 map-to-frame 配准（tracking/odometry）。
  - 回环检索：NetVLAD 全局描述子召回候选回环。
  - 几何验证：SuperPoint + LightGlue 匹配 + 几何一致性过滤。
  - 后端：GTSAM iSAM2 位姿图增量优化（将回环约束注入图中并优化）。
- 新增可复现的评估与汇总能力：
  - 使用 evo 统一输出 ATE/RPE（含对齐与尺度校正）的 zip 结果与图。
  - 提供自动汇总脚本，将 TUM 与 ICL 的 baseline vs loop+PGO 指标整理成一张表（md/csv）。
- 新增批量运行脚本：支持一键跑 TUM 多序列、ICL 多序列，以及带回环/PGO 的批量评估。

### 改进

- 将“运行 / 评估 / 可视化”拆成脚本化步骤，降低重复手工操作与结果丢失风险；统一输出轨迹文件与 evo 结果文件，便于横向对比。
- 针对多环境运行的稳定性做了工程化处理（例如 headless 绘图、日志落盘、避免 SIGPIPE 等），让长批量任务更可控。

## 涉及文件

- `src/rgbd_lc_slam/harness/run_sequence.py`：单序列 baseline 运行入口（tracking/odometry 输出 + 轨迹落盘）。
- `src/rgbd_lc_slam/harness/run_sequence_pg.py`：单序列 loop+PGO 运行入口（回环检索/验证 + iSAM2 增量优化 + 输出 pg 轨迹）。
- `src/rgbd_lc_slam/loop_closure/*`：回环检索（NetVLAD）、几何验证（SuperPoint+LightGlue）、以及回环约束生成等核心逻辑。
- `src/rgbd_lc_slam/backend/isam2_backend.py`：GTSAM iSAM2 位姿图优化封装。
- `scripts/run_all_tum_splits.sh`、`scripts/run_all_icl_nuim.sh`：批量跑 baseline。
- `scripts/run_all_tum_splits_pg_loop.sh`、`scripts/run_all_icl_nuim_pg_loop.sh`、`scripts/run_recommended_pg_loop.sh`：批量跑 loop+PGO。
- `scripts/summarize_all_results.py`：汇总 TUM+ICL 指标到 md/csv。
- `results/summary_tum_icl_loop_pgo.md`、`results/summary_tum_icl_loop_pgo.csv`：阶段性评估产物（用于快速查看）。

## 验证情况

- 已执行验证：
  - 在 TUM `rgbd_dataset_freiburg1_xyz` 上进行最小验证（前 200 帧）：
    - baseline：`python -m rgbd_lc_slam.harness.run_sequence --seq_dir data/tum_rgbd/rgbd_dataset_freiburg1_xyz --out_dir results/runs/minval/tum_fr1_xyz_baseline --max_frames 200`
    - loop+PGO：`python -m rgbd_lc_slam.harness.run_sequence_pg --seq_dir data/tum_rgbd/rgbd_dataset_freiburg1_xyz --out_dir results/runs/minval/tum_fr1_xyz_pgloop --max_frames 200 --enable_loop --device cpu`
    - APE（align+correct_scale, full）：baseline rmse=0.0575；PGO rmse=0.0670
  - 在 ICL-NUIM `living_room_traj1_frei_png` 上进行最小验证（前 200 帧）：
    - baseline：`python -m rgbd_lc_slam.harness.run_sequence --seq_dir data/icl_nuim/sequences/living_room_traj1_frei_png --out_dir results/runs/minval/icl_lr1_baseline --max_frames 200`
    - loop+PGO：`python -m rgbd_lc_slam.harness.run_sequence_pg --seq_dir data/icl_nuim/sequences/living_room_traj1_frei_png --out_dir results/runs/minval/icl_lr1_pgloop --max_frames 200 --enable_loop --device cpu --exclude_recent 50 --retrieval_top_k 5 --retrieval_min_score 0.97`
    - APE（align+correct_scale, full）：baseline rmse=0.0251；PGO rmse=0.0528
- 验证结果：通过（最小验证场景）。
- 备注：本次验证仅覆盖“跑通链路 + 产出轨迹/指标”，不代表回环一定提升精度；回环约束 gating/鲁棒核策略仍需后续优化。

## 后续行动

- 优化 ATE/RPE 偏大的问题：
  - 回环约束质量控制：增加多阶段 gating（时间/位姿先验、匹配内点率、ICP/photometric 复核）、鲁棒核与开关变量（switchable constraints）。
  - 前端稳健性：改进深度/法向/ICP 代价设计，加入动态物体/低纹理场景的退化检测与降权。
  - 参数与工程：统一 TUM/ICL 的相机内参与深度尺度处理；补充关键参数的配置化与 sweep 脚本。
  - 评估体系：补齐更多序列/更多指标（成功率、回环精确率/召回率、实时性统计）。
