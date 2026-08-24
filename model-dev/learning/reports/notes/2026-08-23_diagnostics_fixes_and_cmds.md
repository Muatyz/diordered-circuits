# 2026-08-23 N=120 诊断修复与执行指令

针对 N=120 新代码运行结果的三个问题做了修复，并提供一份用于 von-Mises（以及
N=120 通用）的 diagnostics config 与逐 run 的执行指令。

## 1. 修复内容

### 1.1 PI error 显示改为 mod 2π（问题 1）

- 现象：全协议（cue → darkness → recue）PI error 图在 recue 段出现 `-2π` 平台。
  这是因为累积误差做了连续 unwrap，而 recue 时视觉锚定把解码拉回 true heading，
  展开的误差停在 `-2π`，看起来像"没跟上"，实际已经重新锚定。
- 修复（`analysis/make_vafidis_figures.py`）：
  - 新增 `_pi_error_plot_data(history)`：优先取已存的
    `pi_error_full_protocol_wrapped`（mod 2π），并返回 `use_circular_axis=True`；
    fallback 依次为 `pi_error_full_protocol`（连续）、绝对圆误差、darkness-only
    `pi_error_release_relative`（补 NaN）。
  - 新增 `_pi_error_peak_plot_data(history, circular)`：peak 误差在 circular 模式
    下也 wrap，与主 trace 对齐。
  - constant-velocity 与 OU 两个 figure job 的 `circular_error_axis` /
    `circular_axis` 由返回的 `use_circular_axis` 决定。
- 效果：recue 段误差折回 0 附近；darkness 段若累积超过 π 也会折回（图上保留
  wrapped 视 + 圆形 y 轴）。数值指标（`darkness_rms_pi_error` 等）仍用 unwrapped，
  不受影响。

### 1.2 attractor 弛豫时长与 fixed-point 判断（问题 2）

- 现象：5 s darkness 不足，近 saddle-node 慢化区在 cue release 后仍漂移，
  endpoint map 的 `D(phi)` 未收敛 → unresolved / nonmonotonic crossing 偏多，
  FP 分类不可信。
- 修复：新 diagnostics config 把 `bump_attractor_duration` 5 s → **20 s**、
  `bump_attractor_sample_interval` 0.1 → **0.2 s**（360 轨迹 × ~101 采样点，
  内存/耗时可控）。20 s 让每条轨迹真正弛豫到 basin，`D(phi)` 成为真实的
  endpoint 位移。

### 1.3 图像文字重叠（问题 3）

- `plot_bump_attractor_pva_trajectories`（`plotting/heading.py`）：
  - figsize 10.4×4.8 → 11.8×5.2；
  - trajectory 标题改为两行（第一行标题，第二行图例说明），fontsize 10；
  - endpoint 角注从右下移到**左下**并加白色半透明 bbox（避免与散点/图例重叠）；
  - legend 加 `framealpha=0.82` 背景 + `bbox_to_anchor` 锚定，避免文字叠在数据上；
  - `savefig(..., bbox_inches="tight")` 防止 suptitle 被裁切。
- `plot_heading_and_pi_error_panels`：两个子图 legend 改为两列 + 半透明背景 +
  锚定在轴上方，避免与 heading 轨迹重叠。

## 2. 新 diagnostics config（按网络规模分离）

两个 config 位于 `configs/diagnostics/`（dev 与 release 均已同步）：

| 文件 | 适用 | `bump_attractor_duration` | `sample_interval` | 慢流形/时间尺度 |
| --- | --- | --- | --- | --- |
| `vafidis_diagnostics_n60.yaml` | **N=60** von-Mises | **5.0**（N=60 5 s 已充分弛豫） | 0.1 | **开** |
| `vafidis_diagnostics_n120.yaml` | **N=120** von-Mises / heterogeneous | **20.0**（近鞍点慢化需更久） | 0.2 | **开** |
| `vafidis_diagnostics_vonmises.yaml` | 旧名（= n120），保留兼容 | 20.0 | 0.2 | 开 |

N=60 与 N=120 均启用：`pva_spectrum_and_visualization`（慢流形/Ramesan）、
`timescale_separation`（切向阈值 3°）、`trajectory_and_fixed_points`（FP 分类）。
slow-manifold 参数（`slow_manifold_speed_floor=0.0005`、
`ramesan_phase_velocity_floor=0.001`）与 N 无关。

> 缓存陷阱（已修复，dev+release 同步）：`run_tests_for_existing_run` 现在要求
> ① 启用慢流形/时间尺度时 trajectory 缓存必须含非空 `autonomous_probe_state`；
> ② 缓存时长必须匹配 `bump_attractor_duration`（5 s 缓存不会服务 20 s 运行）。
> 因此旧 5 s 无 autonomous-state 的缓存会自动失效并重算。

## 3. 逐 run 执行指令

`test_vafidis_toy --run-dir` 会用指定 diagnostics config 在已训练权重上重跑全部
启用诊断并重新出图（`weight_source: best`）。run 位于**仓库根** `runs/`（全分支
共享），以下命令在 `model-release\learning`（pro 环境）执行即可，config 已同步到
release。请分别在独立终端执行。

### 3.1 N=60 · seed 42/43/44 —— slow manifold + 时间尺度（5 s）

```bat
cd d:\codefiles\python\diordered-circuits\model-release\learning
conda activate pro
python -m learning.experiments.test_vafidis_toy ^
  --run-dir ..\..\runs\vafidis_baseline\20260822-234631_vafidis_toy_42 ^
  --diagnostics-config configs\diagnostics\vafidis_diagnostics_n60.yaml
python -m learning.experiments.test_vafidis_toy ^
  --run-dir ..\..\runs\vafidis_baseline\20260822-234739_vafidis_toy_43 ^
  --diagnostics-config configs\diagnostics\vafidis_diagnostics_n60.yaml
python -m learning.experiments.test_vafidis_toy ^
  --run-dir ..\..\runs\vafidis_baseline\20260822-234826_vafidis_toy_44 ^
  --diagnostics-config configs\diagnostics\vafidis_diagnostics_n60.yaml
```

### 3.2 N=120 von Mises / heterogeneous —— 20 s 弛豫 + slow manifold

```bat
python -m learning.experiments.test_vafidis_toy ^
  --run-dir ..\..\runs\vafidis_von_mises\20260823-021229_vafidis_von_mises_42 ^
  --diagnostics-config configs\diagnostics\vafidis_diagnostics_n120.yaml
python -m learning.experiments.test_vafidis_toy ^
  --run-dir ..\..\runs\vafidis_heterogeneous\20260823-021710_vafidis_heterogeneous_42 ^
  --diagnostics-config configs\diagnostics\vafidis_diagnostics_n120.yaml
```

### 3.3 visual anneal（pro-anneal 快照；等 anneal 训练结束后同步快照再跑）

```bat
cd d:\codefiles\python\diordered-circuits\model-release-anneal\learning
conda activate pro-anneal
python -m learning.experiments.test_vafidis_toy ^
  --run-dir runs\vafidis_visual_anneal\20260823-152104_vafidis_visual_anneal_42 ^
  --diagnostics-config configs\diagnostics\vafidis_diagnostics_n120.yaml
```

> anneal 是 N=60，也可用 `n60` config；且快照需先同步新代码
> （`--unlock-training` → `--apply --allow-dirty` → `--lock-training`）。

## 4. 补充建议

1. **heterogeneous 也建议用 20 s**：其近 saddle-node 慢化比 von Mises 更明显
   （之前 75 个 nonmonotonic crossings），20 s 弛豫会显著改善 FP 分类。
2. **先看 `slow_manifold_eta_theta_deg_s` / `spectral_gap_min` /
   `settled_fraction_median`**：这三个量直接回答 quasi-continuous 问题，
   是 von-Mises vs heterogeneous 对比的核心指标。
3. 重跑诊断很耗时（360 轨迹 × 20 s），建议一次跑一个 run、各自独立终端。
4. 若 20 s 仍有个别轨迹未收敛（极慢临界慢化），`classify_endpoint_map_fixed_points`
   会如实标记 unresolved——这是信息而非 bug，解读时注意。
