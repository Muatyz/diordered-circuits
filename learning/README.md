# Vafidis-style predictive local plasticity toy model

`learning/` 用于检验 Vafidis-style 局部 predictive plasticity 能否学习 HD bump、速度驱动的路径积分，以及可解释的 `HD→HD` / `HR→HD` 权重结构。训练只使用局部变量，不包含 backpropagation、全局 loss 或 supervised RNN trainer。

除非特别说明，以下命令都从 `learning/` 目录运行。

## 安装与快速开始

```bash
conda activate random
python -m pip install -e .
```

运行基线模型：

```bash
python -m learning.experiments.run_vafidis_toy --config configs/experiments/vafidis_toy.yaml
```

运行 heterogeneous 模型：

```bash
python -m learning.experiments.run_vafidis_toy --config configs/experiments/vafidis_mammalian_heterogeneous.yaml
```

运行纯 population-mean scaling 实验：

```bash
python -m learning.experiments.run_vafidis_toy --config configs/experiments/vafidis_population_mean_heterogeneous.yaml
```

运行 population-mean scaling 下的 von-Mises 对照：

```bash
python -m learning.experiments.run_vafidis_toy --config configs/experiments/vafidis_population_mean_von_mises.yaml
```

快速检查时可添加 `--no-figures`。已有权重可以独立重测或重画，不需要重新训练：

```bash
python -m learning.experiments.test_vafidis_toy --run-dir runs/vafidis_toy/<run_id>
python -m learning.analysis.make_vafidis_figures --run-dir runs/vafidis_toy/<run_id>
python scripts/inspect_run.py runs/vafidis_toy/<run_id>
```

运行测试：

```bash
python -m pytest -q
```

Windows 临时目录清理失败时可使用：

```bash
python -m pytest -q --basetemp runs/.pytest_tmp
```

## 当前实验配置

| 配置 | 当前用途 |
| --- | --- |
| `configs/experiments/vafidis_toy.yaml` | `N_HD=N_HR=60` 的同质 von-Mises teacher 基线 |
| `configs/experiments/vafidis_mammalian_heterogeneous.yaml` | `N_HD=N_HR=360` 的 Clark-style heterogeneous teacher |
| `configs/experiments/vafidis_population_mean_heterogeneous.yaml` | `N_HD=N_HR=360`、纯 `1/N_pre` scaling 的 heterogeneous teacher |
| `configs/experiments/vafidis_population_mean_von_mises.yaml` | population-mean 新动力学下的同质 von-Mises teacher 对照 |
| `configs/analysis/visual_current_noise_std_comparison.yaml` | visual-current OU noise 强度 sweep |
| `configs/analysis/neuron_count_comparison.yaml` | 基线模型的 neuron-count sweep |
| `configs/analysis/heterogeneous_neuron_count_comparison.yaml` | heterogeneous teacher 下的 neuron-count sweep |
| `configs/analysis/noise_by_neuron_count_comparison.yaml` | noise × neuron-count 网格实验 |
| `configs/analysis/heterogeneous_30_mouse_hyper.yaml` | 30 只独立模拟小鼠的 `N=360` Clark-style 分析 |

分析 preset 的统一入口为：

```bash
python -m learning.experiments.run_attractor_robustness --robustness-config configs/analysis/<preset>.yaml
```

常用覆盖参数包括 `--train-duration`、`--noise-stds`、`--neuron-counts`、`--seed-offsets`、`--no-progress`、`--no-training-progress` 和 `--no-skip-existing-runs`。实际参数和输出目录以 YAML 为准。

## 模型约定

每一步按以下顺序更新：

```text
true heading / velocity
→ visual and velocity inputs
→ HR dynamics
→ HD distal and proximal dynamics
→ firing rate and local prediction error
→ PSP traces
→ plastic weights（仅训练阶段）
```

重要约定：

- 训练阶段提供 visual teacher 并更新权重；测试阶段冻结权重，visual 只用于 cue 或 re-cue。
- 数值积分使用统一 `dt` 的 Euler baseline。
- HD 使用 paired geometry：相邻的一对细胞共享 angular preference，因此 `N_HD=360` 对应 180 个不同的预设角度，但调谐曲线分析仍包含全部 360 个 HD 神经元。
- activity heatmap 在角度轴上合并 paired partners；weight matrix 则按经验 COM 排序后的 neuron ID 绘制，坐标是均匀的离散神经元编号。
- heading 解码以 population-vector average / circular COM 为主；peak decode 只作为诊断。
- 新 scaling 配置把 plastic `W` 解释为 intensive kernels：HD→HD 除以 `N_HD`，LHR/RHR→HD 分别除以各自 wing size；固定一对一 HD→HR 和逐细胞外部电流不缩放。该实验不使用 `N_ref`，也不复用旧模型权重。
- frozen-weight Phase 1A 可在该配置上设 `simulation.plasticity_enabled: false`，并把两项初始化 mode 改为 `local_kernel`；默认配置为从零权重在线学习的 Phase 1B。

## Heterogeneous visual teacher

`vafidis_mammalian_heterogeneous.yaml` 使用 wrapped-Gaussian covariance 和 normalized softplus 生成每个神经元固定、可非对称或多峰的视觉 profile。每条曲线按 circular COM 对齐到模型的 HD preference。

当前 teacher 使用 `unit_angular_mean`：每条生成曲线的角度均值为 1，再统一乘以 shared amplitude。这样总视觉驱动不依赖种群中偶然出现的最大峰，同时保留神经元之间的峰高差异。若需要严格控制每个神经元的峰值，可改用 `per_neuron_peak`。不再支持依赖 population size 和 seed 的 population-global peak 归一化。

标准 heterogeneous run 会额外输出：

- `figures/activity/heterogeneous_visual_input_profiles.png`：4 × 4 sampled profiles。
- `hd_tuning_history.npz`：冻结权重后的 heading sweep、settling 和 convergence 信息。
- `hd_tuning_com_aligned.npz`：全部 HD 神经元的 COM、整数 bin 对齐结果、有效性 mask，以及 peak / unit-mean 两种汇总。
- `figures/activity/single_neuron_hd_tuning_curves.png` 和 COM-aligned mean/std 图。

COM-aligned 主图先将每个神经元除以自己的最大 firing rate，再按各自 COM 平移；浅线显示单神经元，粗黑实线显示全体神经元均值。接近静默的细胞保留在 peak-normalized 全体统计中并记录 validity 信息，不会再导致整次绘图中止。

## 30-mouse heterogeneous hyper experiment

当前 preset 使用 seed offsets `0..29`，生成 30 个相互独立的 `N_HD=N_HR=360` 网络。每个 seed 独立采样初始化、OU 训练轨迹和 heterogeneous visual profiles。运行命令：

```bash
python -m learning.experiments.run_attractor_robustness --robustness-config configs/analysis/heterogeneous_30_mouse_hyper.yaml
```

### Cross-mouse COM-aligned tuning

每只小鼠先在其全部有效神经元上计算 peak-normalized COM-aligned mean/std：

- 浅灰线：每只小鼠的 neuron-level mean 或 std。
- 粗黑线：30 只小鼠曲线的等权平均。

主要输出：

```text
heterogeneous_cross_mouse_tuning_n360_per_neuron_peak.npz
figures/heterogeneous_cross_mouse_tuning_n360_per_neuron_peak.png
```

### Clark-style Figure 4 A/B/C

- A：每只小鼠全部有效 HD tuning COM 的黑点；同时给出逐鼠 Kuiper uniformity test，并对 30 个 p-value 做 Benjamini–Hochberg correction。
- B：与每只小鼠有效神经元数量匹配的 uniform circular null。
- C：对 mouse `1/10/20/30` 使用嵌套的 `N_sub={5,10,20,40,80,120,240,360}` 子集，计算 Clark Eq. 4 的 uncentered two-point correlation：`C = R.T @ R / N_sub`。

每个 C panel 的 `epsilon_circ` 是 correlation matrix 到其最佳 circulant projection 的相对 Frobenius 距离；越接近 0，矩阵越接近平移对称。横纵轴是均匀 heading sample，而不是 COM-sorted neuron ID，因此即使经验 COM 不完全均匀，heatmap 的角度网格仍然正确；COM 覆盖不足会表现为较大的 sampling fluctuation 或 `epsilon_circ`。

输出包括：

```text
figures/heterogeneous_clark_figure4_abc.png
heterogeneous_clark_figure4_statistics.npz
heterogeneous_clark_figure4_com_uniformity.csv
heterogeneous_clark_figure4_circulant_error.csv
heterogeneous_clark_figure4_summary.json
```

## 测试与判读

标准 frozen-weight protocols 覆盖：

- cue 后的 bump maintenance；
- 恒速和 OU 速度下的 darkness path integration；
- velocity gain 与近零速度 operating range；
- zero-velocity bump diffusion ensemble；
- 均匀初始角的 zero-input attractor trajectory map；该测试并列使用
  PVA、peak neuron 和 Clark Eq. 6 的未中心化 overlap decoder，overlap 的
  target manifold 来自同一次 frozen-weight HD tuning sweep；
- frozen-weight HD tuning sweep；
- weight structure、谱和训练过程诊断。

判读时优先检查 `test_metrics.json`、收敛比例、PVA strength、darkness velocity-tracking error 和 diffusion，而不是只看一张活动图或全速度范围的线性 gain。当前模型仍是机制诊断用 toy model；增加神经元数量前，应先确认单鼠网络形成单一、可移动且在 darkness 中稳定的 bump，并检查结果是否在多个 seed 上一致。

### Zero-input attractor landscape

所有 `configs/experiments/*.yaml` 单小鼠配置默认启用同一套 frozen-weight
screening protocol：在 `[-pi, pi)` 上设置 36 个均匀 cue 位置，每个位置先用
4 s stationary visual cue 初始化 bump，再关闭 visual input 和 velocity input，
每 0.25 s 记录一次。普通 screening 使用 30 s darkness；当前 N=120 的
`vafidis_population_mean_von_mises.yaml` 使用 120 s，因为已有结果在 30 s 时仍未
完全到达 basin 中心。整个测试冻结权重；它不参与训练，也不需要为了重测而重新
训练。增加起点数只提高角度分辨率，增加 darkness duration 才能确认慢收敛终点。

同一条 HD trajectory 使用三个 decoder：

- PVA / circular COM；
- strongest peak neuron；
- Clark Eq. 6 的未中心化 overlap order parameter：
  `m(theta,t) = phi_star(theta)^T phi(t) / N`，取 `argmax_theta m(theta,t)`；其中
  `phi_star` 来自同一次 frozen-weight HD tuning sweep，而不是视觉输入模板。

输出数据为 `bump_attractor_trajectory_history.npz`，主图为
`figures/heading/bump_attractor_decoder_trajectories.png`。左列显示所有 darkness
trajectory；理想连续环中每条线都应从一开始便近似水平。右列是 initial cue 到
final decoded angle 的 endpoint map；理想结果接近恒等线。若多个初始角收缩到
少数水平带，则网络形成的是离散 basin / pinned bumps，而不是连续吸引子流形。
darkness observation window 是逐模型 config 参数，而不是绘图硬编码：

```yaml
tests:
  bump_attractor_trajectory_enabled: true
  bump_attractor_initial_conditions: 36
  bump_attractor_duration: 120.0       # time in darkness [s]
  bump_attractor_cue_duration: 4.0
  bump_attractor_sample_interval: 0.25
```

endpoint map 始终使用 `bump_attractor_duration` 对应的最后一个样本，图标题同时写出
实际 `T_dark`。若 endpoint 尚未形成稳定水平带，可提高 duration；若需要更准确地
定位 basin boundary，则提高 initial conditions。计算量近似正比于两者的乘积。
PVA strength 或 bump contrast 较高只能说明 bump 仍存在，不能证明其位置方向
具有连续中性模；同理，收缩完成后的 late-time drift 接近零也不能单独作为证据。

`test_metrics.json` 分别记录三个 decoder 的 initial alignment、trajectory
flatness、final/max displacement、defined fraction、5/10 degree stability fraction
和 late-time drift，并给出 PVA/peak 相对 overlap 的 disagreement。三种 decoder
若得到相同 endpoint basin，可排除单一 decoder mismatch。step-level 进度条显示
当前起点、初始角、cue/darkness phase、steps/s 和预计剩余时间。

使用 `--run-dir` 重测时，程序读取该 run 内的 `config_resolved.yaml`，不会读取
当前 `configs/experiments/*.yaml`。在该功能加入之前生成的旧 run 必须把五个
`bump_attractor_*` 字段同步到其 resolved config；否则该测试默认关闭并只写出
空 history。

### Clark-style time-scale separation

`vafidis_population_mean_von_mises.yaml` 还启用了独立的 frozen-weight
time-scale separation assay。它实现的是 Clark Figure 3E/F 对
“fast attractive normal flow + slow tangential flow”的 operational test：

- `T_perp`：从 12 个均匀流形位置出发，在 HD distal-current space 中施加
  RMS 为 `0.025/0.05/0.1` 的随机扰动，并去掉局部 tangent 分量；随后在无视觉、
  无速度输入下测量到 closed piecewise-linear target manifold 的距离。距离使用
  full-state L2 / `sqrt(N)`，并由早期峰值衰减到 late floor 上方 `1/e` 的时间定义
  normal relaxation time。
- `T_parallel`：复用 long-darkness Clark-overlap trajectory，以绝对位移首次达到
  10 degree 的时间定义 tangential first-passage time；未在配置的 observation
  window 内通过的轨迹按
  right-censored data 处理。
- 保守判据使用 `T_parallel(p10) / T_perp(p90) >= 10`，同时要求至少 90% 的
  normal perturbations 完成 e-fold recovery 且最终绝对距离降到峰值的 `1/e`
  以内。若 p10 本身仍被 observation window censor，ratio 会明确标记为 lower
  bound，而不会伪装成精确值。

原始结果保存在 `timescale_separation_history.npz`，汇总指标写入
`test_metrics.json`，诊断图为
`figures/heading/timescale_separation_diagnostics.png`。该判据检验的是有限时间内的
quasi-continuous dynamics，不等同于证明存在严格的零 Jacobian 特征值；离散 basin
仍应结合 zero-input endpoint map 一起判断。

对已有 trained run 重测不需要重新训练，但 `--run-dir` 使用 run 内的
resolved test config。旧 run 可直接使用显式 override，无需手工修改训练 config：

```bash
python -m learning.experiments.test_vafidis_toy --run-dir "runs/vafidis_population_mean_von_mises/<run_id>" --bump-attractor-duration 120 --enable-timescale-separation
```

首次 override 后会写出 `test_config_resolved.yaml`；后续重测优先复用它。命令会
重新生成 HD tuning current manifold、指定时长的 trajectory 和上述测试输出，但不会
更新训练权重。

### Constant-velocity sweep and dense-probe phase flow

`vafidis_population_mean_von_mises.yaml` 还实现了 Noorman et al., *Maintaining
and updating accurate internal representations of continuous variables with a
handful of neurons* Figure 2f 对应的 frozen-weight constant-velocity test。它与
普通 velocity-gain curve 的区别是保留完整的 bump position trajectory
`x(t)-x(0)`，从而直接区分：

- 低速输入下仍被 discrete basin 锁定；
- 临界速度附近在 basin 之间发生 stick--slip 跳跃；
- 高于 depinning threshold 后持续沿同一方向滑动。

默认输入为
`v = 0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 1.4, 2.0 rad/s`，
重点加密 depinning 前后的低速区间。全部速度 trial 先由同一个
`0 rad` visual cue 将 bump 锚定 4 s，再从同一网络状态的副本出发，在 30 s
darkness 中分别接受恒定正速度。这样所有曲线严格共享同一个 `x(0)`，可以直接
比较斜率和沿途的周期性加速--减速。

输出为：

```text
velocity_trajectory_sweep_history.npz
figures/heading/velocity_trajectory_sweep.png
velocity_phase_flow_summary.npz
figures/heading/velocity_actual_fp_basin_rings.png
figures/heading/velocity_dense_probe_trajectories.png
figures/heading/velocity_phase_flow_diagnostics.png
```

该图已经在 `vafidis_toy.yaml` 和
`vafidis_population_mean_von_mises.yaml` 中默认启用；一次新训练完成后会随其它默认图
自动生成。schema 仍保持 opt-in，以免使用 mammalian base config 的多小鼠 hyper
实验为每只小鼠自动执行这项 `30 s x 13 velocities` 的额外测试。

要在已有训练权重上补跑稠密 phase-flow probe（不重新训练）：

```bash
python -m learning.experiments.test_vafidis_toy --run-dir "runs/<experiment>/<run_id>" --enable-velocity-phase-flow --velocity-phase-flow-probes 360
```

旧 run 的 resolved config 不包含新的低速网格。无需重训，可显式覆盖冻结权重测试速度：

```bash
python -m learning.experiments.test_vafidis_toy --run-dir "runs/<experiment>/<run_id>" --velocity-sweep-values "0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.4,2.0" --velocity-phase-flow-values "0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6" --velocity-phase-flow-probes 360
```

已有 phase-flow history 时，可以只重画这三张诊断图，避免重新渲染其它图：

```bash
python -m learning.analysis.make_vafidis_figures --run-dir "runs/<experiment>/<run_id>" --phase-flow-only
```

图 A 将所有输入速度的 unwrapped PVA bump position 画在同一坐标轴：理想连续
吸引子产生斜率恒定的直线；存在离散 basin 时会出现周期性的变速、停顿或跳跃。
图 B 绘制 `x(t)-x(0)-vt`，去掉大尺度线性位移，使较小的“过山车”波动更容易辨认。
图 A/B 只按输入速度连续着色，不再用 `x_ddot/x_dot` 将单条轨迹片段命名为
stable/unstable-FP region；后者在低速、转折点和噪声附近数值不稳定，也不能证明真实
FP 的存在。图 C 显示 input-output velocity，并用 Clark-overlap decoder 交叉验证。
FP 分析不再把 `dF_v/dtheta` 的同号区间和 `F_v=0` 的根混为同一种 region。默认对
`0..0.6 rad/s` 的 9 个低速点分别运行 360 个均匀起点、2 s 的独立 frozen-weight
probe，并每 0.02 s 记录 PVA。所有实际观测到的 `(theta, dot(theta))` 按 360 个角度
bin 汇总，中位数给出离散流速场；空 bin 只做周期线性插值，随后使用 5-bin 圆周移动
平均。该流程不拟合 Fourier 基、不使用 ridge 或 root-confidence 外推。

`velocity_dense_probe_trajectories.png` 显示全部独立 probe 的原始轨迹，以便先检查采样
是否覆盖全圆以及是否存在 cue-release 瞬态。`velocity_actual_fp_basin_rings.png` 只报告
离散流场中发生符号变化的真实动力学对象：正到负的零点为 stable FP，负到正的零点为
unstable separatrix；每个 stable basin 是相邻 unstable FP 之间的圆弧。黑色实心点为
stable FP，黑叉为 unstable FP。若 `F_v(theta)` 在全圆同号，则表示已经 depin，整环
显示为灰色，不再外推并不存在的 FP 或 governing region。

`velocity_phase_flow_diagnostics.png` 左列同时显示每个 bin 的原始速度中位数、局部平滑
后的 `F_v(theta)` 与实际根；右列显示圆周差分得到的 `F_v'(theta)`。对于 `v != 0`，
还从同一批轨迹计算经验量 `median(ddot(theta)/dot(theta))`。若一维自治约化
`dot(theta)=F_v(theta)` 成立，则二者应满足
`ddot(theta)/dot(theta)=F_v'(theta)`；不一致提示隐藏的 bump-shape/HR 状态、瞬态记忆
或 decoder 误差。低速样本会按速度阈值剔除，`v=0` 不计算该比值，且该比值不参与
FP 或 basin 的定义。

旧 D 图中的 success fraction、stall fraction、speed-linearity 和 depinning threshold
仍保存在 NPZ/JSON，没有丢弃定量结果。原始 NPZ 同时保留
PVA、literal peak 和 overlap 三种 decode。主图采用连续 PVA 而不是 literal
argmax peak，是为了避免把有限神经元造成的 `2*pi/N` 量化台阶误认为 basin
stick--slip。operational depinning
threshold 定义为第一个满足以下条件的输入速度：至少 90% 初始角达到
`decoded/input >= 0.5`，且各自的 stall fraction 不超过 0.2。该阈值依赖测试时长
和判定标准，因此应与原始 trajectory 一起报告，不能视为模型的解析临界点。

已有 trained run 无需重新训练：

```bash
python -m learning.experiments.test_vafidis_toy --run-dir "runs/vafidis_population_mean_von_mises/<run_id>" --enable-velocity-trajectory-sweep
```

## 输出结构

单次训练通常写入：

```text
runs/<experiment>/<run_id>/
  config_resolved.yaml
  training_history.npz
  weight_history.npz
  trained_weights.npz
  *_history.npz
  hd_tuning_com_aligned.npz
  test_metrics.json
  figures/
```

robustness / hyper 实验写入 `reports/attractor_robustness/<preset>/<timestamped_run>/`，包含逐 run summary、aggregate CSV/JSON、复用的单鼠 run 和汇总图。`skip_existing_runs: true` 只复用所需产物完整的 run。

## 代码结构

```text
src/learning/models/vafidis_toy.py                 核心状态与单步动力学
src/learning/stimuli/visual.py                     visual teacher 与 visual noise
src/learning/experiments/run_vafidis_toy.py        单次训练、测试和保存
src/learning/experiments/run_attractor_robustness.py  sweep 与跨鼠分析
src/learning/analysis/make_vafidis_figures.py       单次 run 的汇总和绘图
tests/                                               配置、动力学、指标和绘图测试
```

开发记录和尚未完成的实验问题保留在 `.todo/TODO.md`；README 只描述当前可运行功能。
