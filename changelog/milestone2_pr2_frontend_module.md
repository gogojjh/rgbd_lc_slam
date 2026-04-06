# Milestone 2 (PR2) — Frontend 模块化（Tracking 从 harness 抽到 frontend）

- 分支：`milestone2-frontend-module`
- 目标：在不改变 Stage-1 算法链路（map-to-frame ICP tracking + loop closure + iSAM2 PGO）的前提下，将 tracking 前端封装为可复用模块，harness 只负责数据 IO / 调用 / 输出。

## 改动内容

### 1) 新增前端模块（可复用）

- `src/rgbd_lc_slam/frontend/types.py`
  - `RGBDTrackingConfig`：voxel/submap_k/keyframe 阈值等参数
  - `RGBDFrame`：单帧数据容器（fid/stamp/rgb/depth_m）
  - `TrackingResult`：输出（Twc/is_keyframe/tracking_ms）

- `src/rgbd_lc_slam/frontend/rgbd_icp_frontend.py`
  - `RGBDICPFrontend`：封装 submap + keyframe + ICP tracking（尽量贴近原 harness 行为）

- `src/rgbd_lc_slam/frontend/submap.py` / `keyframe.py`
  - submap 构建与 keyframe 判定的前端内部实现

### 2) harness 脚本变薄

- `src/rgbd_lc_slam/harness/run_sequence.py`
  - 不再直接操作 Open3D ICP/pointcloud/submap；改为：加载图像 → 构造 `RGBDFrame` → 调用 `RGBDICPFrontend` → 输出轨迹

- `src/rgbd_lc_slam/harness/run_sequence_pg.py`
  - tracking 由 `RGBDICPFrontend` 负责
  - backend(iSAM2) 与 loop closure 模块仍在 harness 里（保持 Stage-1 pipeline 结构）

### 3) 稳定性修复：避免 LightGlue 首次运行阻塞下载

- `src/rgbd_lc_slam/loop_closure/matching.py`
  - 新增 `_has_superpoint_weights()`：若 SuperPoint 权重未在 cache 中，则**直接使用 ORB fallback**
  - 目的：避免在某些环境里 LightGlue 触发权重下载（或卡在动态库/ABI）导致 `run_sequence_pg.py --enable_loop` 挂住。

## 最小验证（回归）

> 说明：harness 默认不会自动复制 `groundtruth.txt` 到输出目录；ATE 计算前我手动执行了 `cp .../groundtruth.txt $OUT/traj_gt_tum.txt`。

### TUM: `rgbd_dataset_freiburg1_xyz` (200 frames)

- baseline：
  - cmd：`python -m rgbd_lc_slam.harness.run_sequence --seq_dir data/tum_rgbd/rgbd_dataset_freiburg1_xyz --out_dir results/runs/minval_m2/tum_fr1_xyz_baseline --max_frames 200`
  - APE rmse (align+correct_scale, full)：**0.0226**

- loop+PGO：
  - cmd：`python -m rgbd_lc_slam.harness.run_sequence_pg --seq_dir data/tum_rgbd/rgbd_dataset_freiburg1_xyz --out_dir results/runs/minval_m2/tum_fr1_xyz_pgloop_200 --max_frames 200 --enable_loop --device cpu`
  - APE rmse (tracking)：**0.0226**
  - APE rmse (PGO)：**0.0366**

### ICL-NUIM: `living_room_traj1_frei_png` (200 frames)

- baseline：
  - cmd：`python -m rgbd_lc_slam.harness.run_sequence --seq_dir data/icl_nuim/sequences/living_room_traj1_frei_png --out_dir results/runs/minval_m2/icl_lr1_baseline --max_frames 200`
  - APE rmse：**0.00212**

- loop+PGO：
  - cmd：`python -m rgbd_lc_slam.harness.run_sequence_pg --seq_dir data/icl_nuim/sequences/living_room_traj1_frei_png --out_dir results/runs/minval_m2/icl_lr1_pgloop_200 --max_frames 200 --enable_loop --device cpu --exclude_recent 50 --retrieval_top_k 5 --retrieval_min_score 0.97`
  - APE rmse (tracking)：**0.00212**
  - APE rmse (PGO)：**0.00230**

## 备注

- 本 milestone 的目标是“模块化 + 跑通回归”，不承诺 PGO 一定提升 ATE。
- loop closure 依赖 lightglue/superpoint 权重；为了避免在无权重/不稳定环境里阻塞，matcher 会在缺权重时自动退化到 ORB。
