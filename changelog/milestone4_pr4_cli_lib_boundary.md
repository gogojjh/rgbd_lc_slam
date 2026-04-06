# Milestone 4 (PR4) — 核心库 vs 实验脚本边界进一步清晰

- 分支：`milestone4-cli-lib-boundary`

## 目标
- 把“可复用 IO/工具”从 `harness/` 迁到 `rgbd_lc_slam.io/`，让 harness 更像纯编排层。
- 增加 `rgbd_lc_slam.cli` 作为更稳定的命令行入口（以后 harness 可以再继续变薄/甚至内部化）。

## 变更
### 1) IO helper 迁移
- `src/rgbd_lc_slam/harness/common_rgbd.py` → `src/rgbd_lc_slam/io/rgbd_io.py`
- 新增 `rgbd_lc_slam.io.__init__` 导出：`default_intrinsics/is_icl_nuim/load_rgb_depth/rgbd_from_arrays`
- 更新引用：
  - `harness/run_sequence.py` / `harness/run_sequence_pg.py`
  - `frontend/rgbd_icp_frontend.py` 中注释同步

### 2) 新增 cli 包
- `src/rgbd_lc_slam/cli/run_sequence.py`
- `src/rgbd_lc_slam/cli/run_sequence_pg.py`

> 当前 cli 只是薄封装，仍委托到 harness 的 `main()`；后续可以把 argparse + runner 进一步抽到 core library。

## 验证
```bash
conda run -n rgbd_lc_slam python -m pytest -q
# 7 passed
```

## 兼容性
- `python -m rgbd_lc_slam.harness.run_sequence*` 仍可用。
- 新增 `python -m rgbd_lc_slam.cli.run_sequence*` 入口（后续建议 scripts 逐步切换到 cli）。
