# Slow-manifold 与 timescale-separation 诊断：测定原理与解读指南

Date: 2026-08-23
Status: 代码已重做并通过测试（`model-dev/learning`）；两组诊断在实验中仍默认关闭，
待新 N=120 von-Mises/heterogeneous 训练完成后按文末清单启用。

## 1. 这些诊断回答什么问题

你的核心问题是：Vafidis 局部学习规则 + Clark-style heterogeneous visual cue
训练出的网络，其 zero-input 自治动力学是

- (a) 一个**连续吸引子流形**（Sagodi/Clark 意义上的 slow manifold：法向收缩、切向缓慢漂移），
- (b) 还是**一堆离散吸引子**（24 个深浅不一的势阱），
- (c) 还是介于两者之间的 **ghost/quasi-continuous attractor**（接近连续、但有小势垒）。

slow-manifold 与 timescale-separation 诊断就是为区分这三者而设计的，它们测的是
**同一件事的不同投影**：

| 判据（Sagodi notebook §4–§7） | 测量量 | 连续吸引子 | 离散 FP | 你的 N=120 hetero |
| --- | --- | --- | --- | --- |
| 法向收缩 + 切向慢 | Jacobian 谱隙 $\lambda_1-\lambda_2$ | $\lambda_1\approx0$，谱隙大 | $\lambda_1<0$ 随位置振荡 | 待测 |
| 慢流形上切向速度小 | 最大切向相位流 $\eta=\max|\dot\theta_\parallel|$ | $\eta\to0$ | $\eta=$ 势垒决定 | 待测 |
| 记忆误差上界 | $e(T)\le T\|\phi\|_\infty$ | 小 | $\pi/K$ | 待测 |
| FP 多而密 | FP 数量/间距/basin 熵 | 无离散 FP | $K$ 个稳定 FP | ~24 个 |
| 时间尺度分离 | normal vs tangential 弛豫时间比 | $\gg 1$ | 大（势垒高） | 待测 |

## 2. 三个诊断各自测什么

### 2.1 Ramesan slow-ring geometry（`analyze_ramesan_firing_rate_geometry`）

- **输入**：360 个均匀 cue 角度 → 1s 强 cue（amplitude=24）→ release 后的完整
  600 维 Markov 态 $x=[r_{HD\to HR}^{LP}, r_{HR}, i_{HD,d}, v_{HD,d}, v_{HD,a}]$。
- **PCA**：对 block-standardized 的完整状态做 SVD。PC1–3 只是可视化（cue 定义的环），
  **动力学判定不用 PCA**。
- **q(x)**：$q=\frac12\|F_{\Delta t}(x)\|^2$，$F=(G_{\Delta t}(x)-x)/\Delta t$ 是冻结权重的
  零输入离散流。q 小 = 接近不动点。**绝对值无意义**（由 HD→HR 快差主导，~1e7），
  所以慢点判定必须用相对/物理阈值（见 §3）。
- **Jacobian**：在慢候选点上解析计算全状态 Jacobian，报告 $\lambda_1,\lambda_2$ 与
  $\lambda_1-\lambda_2$。谱隙大且 $\lambda_1$ 接近 0 = 接近中性（连续）；$\lambda_1$ 显著
  <0 = 稳定 FP。

### 2.2 Slow-ring / slow-manifold 候选（`analyze_slow_manifold_candidates`）

- **候选集**：每条 darkness 轨迹中 $\|F\|\le$ 阈值的点（阈值 = 相对 1e-3×max 与
  物理 floor 取 min，见 §3）。候选点覆盖的角度区间 = "慢集"覆盖的环。
- **spline 拟合**：$x=c(\theta)$ 周期三次样条 → 流形上的法向流、切向流
  $\dot\theta_\parallel$、有效势垒、FP 根（$\dot\theta_\parallel=0$，稳定/不稳定交替）。
- **读出**：`slow_manifold_eta_theta_deg_s`（最大切向相位流，记忆误差上界来源）、
  `slow_manifold_spectral_gap_min`、`slow_mode_tangent_alignment_median`、
  `slow_manifold_basin_entropy`（basin 大小分布熵，均匀环=高熵）、FP 数量。
- **关键修正在于候选集定义**：旧实现只按相对阈值筛点，而轨迹 max 由弛豫瞬态
  （~7e3/s）决定，1e-3×max≈7/s 会把弛豫中期的点也算"慢"，候选偏 late-time basin，
  `angular_support_fraction` 量的是 basin 覆盖而非 manifold 覆盖。

### 2.3 Phase landscape（`analyze_ramesan_phase_landscape`）

- **输入**：时间上均匀子采样的轨迹态（正式配置 16384 点）。
- **settled/moving 分解**（本次重做）：解码 PVA 相位速度 $v=\mathrm{wrap}(\Delta\theta)/\Delta t$
  在 0.1s 采样网格上，把帧分为
  - **settled**：$|v|<v_{floor}$（1e-3 rad/s ≈ 0.06°/s）→ 该相位是吸引子（时间占比高）；
  - **moving**：$|v|\ge v_{floor}$ → bump 实际跨越该区间的漂移率。
  - 旧实现把两者混在 within-bin median 里：FP 处连续 0 帧 + 移动时单帧跳变 → 看起来像
    "非保守场"。分解后：吸引子 bin 的 settled fraction ≈ 1、moving 速度小；
    过渡 bin 的 settled ≈ 0、moving 速度大。`framesan_phase_settled_fraction_median`
    与 `moving_velocity_median_deg_s` 是直接可比的流形平坦度指标。
- 有效势与根仍基于 smoothing 后的 moving 速度场。

### 2.4 Timescale separation（`run_timescale_separation_test`）

- **normal（法向）**：Clark Fig.3 风格。cue 弛豫后，在 HD distal current 空间沿
  环法向加随机扰动（RMS=0.025/0.05/0.1），测回到最近闭合流形的 e-folding 时间
  $t_{\perp}$。
- **tangential（切向）**：Clark-overlap 位移首达阈值时间 $t_\parallel$（bump 漂移
  $\ge$ 阈值）。
- **判据**：conservative ratio $=t_{\parallel}^{p10}/t_{\perp}^{p90}$。$>10$ 且
  normal 恢复 ≥90% → 时间尺度分离成立。
- **本次修改**：阈值从 10° 降到 3°。因为 N=120 hetero 的稳定 FP basin 宽度中位
  ~10°，10° 阈值几乎立即被越过，$t_\parallel$ 不再反映 basin 内漂移。3° 位于
  cue-release 对齐误差（~1°）之上、basin 宽度之下。

## 3. 本次代码修改摘要（model-dev）

1. `slow_manifold.py::select_slow_candidate_indices`：新增 `speed_floor`（物理速度
   下限，rad/s）与 `time`（时间均匀子采样）。阈值 =
   `min(speed_fraction*max, speed_floor)`；超预算时按时间均匀重采样，避免候选
   全部挤在 late-time basin。
2. `slow_manifold.py::analyze_ramesan_phase_landscape`：新增 `phase_velocity_floor`
   参数与 settled/moving 分解；smoothing、有效势、根查找改用 moving 速度场。
3. `run_vafidis_toy.py`：候选收集传入 `slow_manifold_speed_floor` 与 `time`；
   phase landscape 传入 `ramesan_phase_velocity_floor`；metrics 记录实际阈值与
   floor。
4. `schema.py`：新增 `slow_manifold_speed_floor`、`ramesan_phase_velocity_floor`。
5. `plotting/slow_manifold.py`：面板 B 显示 settled-fraction（青色）+ moving
   IQR/中值（橙色），旧 history 自动回退到 raw median。
6. `configs/diagnostics/vafidis_diagnostics.yaml`（dev）：
   `slow_manifold_speed_floor: 0.0005`、`ramesan_phase_velocity_floor: 0.001`、
   `timescale_separation_tangential_threshold_deg: 3.0`。
7. 测试：新增 floor/time 子采样单测；`test_slow_manifold.py`、
   `test_slow_manifold_pipeline.py`、smoke、plotting 全部通过（dev 196 passed；
   `test_config_defaults.py` 的 6 个失败与 `test_original_basin_test.py`、
   `test_plotting_activity.py` 为预存问题，与本次修改无关）。

## 4. 启用清单（等新训练完成后）

1. 等 N=120 von-Mises 与 heterogeneous 训练结束（正在运行）。
2. 用 `test_vafidis_toy --run-dir ...` 在 best weights 上跑
   `--diagnostics-config`（先保持 slow_manifold/timescale 关闭，确认常规诊断稳定）。
3. 对比两网络的：`slow_manifold_eta_theta_deg_s`、`spectral_gap_min`、
   `slow_mode_tangent_alignment_median`、`basin_entropy`、FP 数、
   `ramesan_phase_settled_fraction_median`、`moving_velocity_median_deg_s`。
   若 heterogeneous 的 eta 更小、谱隙更大、settled 更均匀 → 支持
   quasi-continuous 结论；否则 heterogeneous 只是"cue transfer 连续化 + 积分器
   更好"，自治流形仍是离散环。
4. 确认后才在 diagnostics YAML 中打开
   `pva_spectrum_and_visualization` / `slow_manifold` / `timescale_separation`。

## 5. 已知注意点

- `slow_manifold_speed_floor` 必须与 `bump_attractor_duration` 匹配：floor 太严而
  时长太短时，临界慢化轨迹（近 saddle-node 对）达不到 floor → 候选覆盖出现缺口。
  这个缺口本身有信息量（= 慢化区域），但解读时不要误判为"流形不完整"。
- `ramesan_phase_velocity_floor` 的取值应高于 FP 处 PVA 解码的数值抖动
  （bump 诊断 drift_velocity p95≈0.47°/s = 0.0082 rad/s），低于真正的漂移速率。
  正式配置 1e-3 rad/s 对 5s 时长是安全的。
