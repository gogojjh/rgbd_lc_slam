# Milestone 3 (PR3) — tests + 最小回归保护 + CI

- 分支：`milestone3-tests-ci`

## 目标
- 建立轻量回归保护：不下载数据集、不跑长序列，在几分钟内完成。
- CI 上允许缺失重依赖（open3d/torch/cv2）时 **自动 skip** 对应测试，而不是 fail。

## 变更
### 1) 新增/增强 pytest tests
- `tests/test_trajectory_io.py`
  - 覆盖 `write_tum_trajectory()` 输出格式：每行 8 列、可 parse
- `tests/test_tum_associate.py`
  - 覆盖 `associate_by_time()` 基本配对与 max_dt reject
- `tests/test_keyframe_policy.py`
  - 覆盖 `should_add_keyframe()` 平移/旋转阈值逻辑
- 更新已有 smoke tests：
  - `tests/test_frontend_smoke.py`：open3d 不可用则 skip
  - `tests/test_loop_orb_fallback.py`：torch 不可用则 skip；同时用 tmp_path 隔离 TORCH_HOME

### 2) 增加 GitHub Actions（pytest）
- 新增 `.github/workflows/pytest.yml`
  - `pip install -e .` + `pytest -q`
  - 只装最小依赖（pytest/numpy/scipy），其余重依赖由 tests 自行 skip

## 本地验证
```bash
conda run -n rgbd_lc_slam python -m pytest -q
# 7 passed
```

## 备注
- 如果后续希望 CI 真正覆盖 open3d/torch/lightglue，可以再加一个 job（或 nightly job）专门安装重依赖并跑完整 tests。
