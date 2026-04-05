# 构建过程参考（压缩版）

> 目的：把本仓库从“可运行”推进到“可批量评估 + 可视化 + 可汇总”的最小闭环，并记录关键踩坑与解决方式，便于复现与二次开发。

## 1. 总体目标与阶段划分

- **阶段 1（已完成）**：传统 RGB-D SLAM 基线 + 回环 + PGO + 批量评估与可视化。
- **输出要求**：
  - 在 **TUM RGB-D** 与 **ICL-NUIM** 上批量跑出轨迹；
  - 用 **evo** 生成 ATE/RPE 指标；
  - 汇总对比 baseline vs loop+PGO；
  - 生成轨迹对比可视化（对齐后，便于定性分析）。

## 2. 方案概览（Pipeline）

- **前端（Tracking / Odometry）**：Open3D 的 RGB-D 配准（map-to-frame）。
- **回环检索（Retrieval）**：NetVLAD 全局描述子召回候选关键帧。
- **回环几何验证（Verification）**：SuperPoint + LightGlue 特征匹配 + 几何一致性过滤，必要时用 ICP 做 refine。
- **后端（Backend）**：GTSAM iSAM2 增量位姿图优化，将回环约束加入图并优化得到 `traj_est_pg_tum.txt`。
- **评估（Evaluation）**：evo_ape/evo_rpe（`--align --correct_scale`）生成 zip 结果与图（ATE/RPE）。

## 3. 批量运行与结果汇总

- 批量跑基线与回环版本：
  - `scripts/run_all_tum_splits.sh` / `scripts/run_all_tum_splits_pg_loop.sh`
  - `scripts/run_all_icl_nuim.sh` / `scripts/run_all_icl_nuim_pg_loop.sh`
- 汇总脚本：
  - `scripts/summarize_all_results.py` 统一读取 `evo_ape*.zip` / `evo_rpe*.zip` 的 `stats.json`，输出 md/csv 总表。

## 4. 关键可视化产物

- ATE/RPE 柱状对比图（baseline vs loop+PGO）。
- 轨迹俯瞰总览图：
  - 对每条序列，将 **GT / baseline / loop+PGO** 进行 **Sim(3) 对齐到 GT** 后叠加绘制。
  - 对齐方式与 evo 保持一致，便于定性观察“是否回环约束引入了错误”。

## 5. 主要问题与观察（来自阶段性结果）

- **ATE/RPE 整体偏大**：说明前端配准与/或回环约束质量仍需提升。
- **回环 + PGO 可能退化**：例如 ICL 的 `office_traj0` 在 loop+PGO 下 ATE 明显变差，指向“错误回环约束进入图”或“缺少鲁棒核/开关变量”问题。

## 6. 工程踩坑与解决（摘要）

- **长批量运行日志丢失 / SIGPIPE**：批量脚本采用落盘日志（run.log），避免父进程退出导致日志管道中断。
- **服务器无显示导致 matplotlib 报错**：强制 `MPLBACKEND=Agg`。
- **环境依赖冲突（cv2/torch/libstdc++）**：在脚本中补充 `LD_LIBRARY_PATH/LD_PRELOAD`（以 conda 环境为准）提升运行稳定性。

## 7. 当前结论

- 已得到一套可复现实验基线，能在 TUM/ICL 上批量运行并产生统一的评估结果与可视化。
- 下一阶段主要工作是：**提高回环约束可靠性与前端稳健性**，降低 ATE/RPE，并减少“PGO 反而退化”的失败案例。
