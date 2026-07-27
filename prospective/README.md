# Emina–Kropff prospective-coding toy model

本目录实现 Emina & Kropff (2026) 的 M0–M3 机制链：移动 Gaussian tutor、带 firing-rate adaptation 和 global divisive inhibition 的竞争层，以及局部 Hebbian + weight-dependent decay 前馈学习。

```text
moving tutor R
  -> feedforward current J @ R
  -> membrane U and adaptation V
  -> divisively normalized rate r
  -> local update dJ/dt = eta * r * (R - alpha * J**beta)
  -> learned local Gaussian connectivity
  -> adaptation-dependent prospective shift
```

当前实现是 externally driven feedforward representation。它尚无 recurrent self-sustenance，因此不能称为 autonomous continuous attractor；M4 以后才会加入递归学习与自维持 moving bump。

## 快速开始

所有命令默认从 `prospective/` 目录运行。项目约定使用 Python 3.11 的 `random` Conda 环境：

```powershell
cd D:\codefiles\python\diordered-circuits\prospective
conda activate random
python -m pip install -e .
```

MP4 动画还需要 `ffmpeg` 二进制。推荐把它安装在同一个 Conda 环境，而不是依赖不透明的系统 PATH：

```powershell
conda install -n random -c conda-forge ffmpeg -y
conda run -n random ffmpeg -version
conda run -n random python -c "from matplotlib import animation; print(animation.writers.is_available('ffmpeg'))"
```

最后一条命令应打印 `True`。如果没有安装 `ffmpeg`，渲染器会回退到 GIF；训练本身不受影响。

运行测试：

```powershell
python -m pytest -q
```

### 1. M1：固定理论 Gaussian 权重，先看神经动力学

该命令使用 Eq. 11–13 构造解析 `J`，并冻结学习：

```powershell
python -m prospective.experiments.run_feedforward `
  --config configs/experiments/fixed_theory_dynamics.yaml `
  --theory-weights
```

它用来检查：

- tutor 是否产生局域的 `U/V/r` bump；
- `V` 是否落后于 `U/r`；
- Euler 时间步是否稳定；
- 在学习前，动力学与解析连接是否自洽。

### 2. M2：从随机前馈权重开始训练

```powershell
python -m prospective.experiments.run_feedforward `
  --config configs/experiments/feedforward_toy.yaml
```

默认 toy 配置为 `N_in = N_c = 32`、`dt = 5 ms`、1000 s 模拟时间。它通常只需数秒完成数值积分，随后生成静态图。命令最后打印 run directory，例如：

```text
runs/feedforward_toy/<YYYYMMDD-HHMMSS_seed11>/
```

若只需要训练和数值文件：

```powershell
python -m prospective.experiments.run_feedforward `
  --config configs/experiments/feedforward_toy.yaml `
  --no-figures
```

查看最新一次 run：

```powershell
$run = Get-ChildItem runs/feedforward_toy -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Get-Content "$($run.FullName)/metrics.json"
```

### 3. M3：冻结已学权重，扫描 adaptation 与速度

把上一步实际输出目录传给 probe：

```powershell
python -m prospective.experiments.run_prospective_probe `
  --run-dir runs/feedforward_toy/<run_id> `
  --config configs/experiments/prospective_shift.yaml
```

结果位于：

```text
<run_dir>/prospective_probe/
  prospective_metrics.csv
  prospective_metrics.json
  representative_trace.npz
  prospective_shift_vs_time.png
  prospective_shift_parameter_map.png
```

`mean_shift = decoded_rate_position - tutor_position`。正速度下正值表示 prospective coding；负速度下，`mean_shift_along_motion > 0` 才表示沿运动方向领先。`m=0`、`v=0` 和负速度都是必要控制。`paper_reset` 条件下，距离边界小于 `2.5 * sigma_R` 的样本会从主 shift 统计中排除。

### 4. 机制动画

动画需要训练时保存逐采样点的 `J`、原始 `delta_J` 和 clipping 状态，因此先运行专用的小规模配置：

```powershell
python -m prospective.experiments.run_feedforward `
  --config configs/experiments/animation_demo.yaml
```

然后对刚才的 run 离线渲染；渲染不会重新训练或修改状态：

```powershell
python -m prospective.animation.render `
  --run-dir runs/animation_demo/<run_id> `
  --config configs/animation/mechanism_long.yaml
```

长动画可分开渲染，避免修改一个镜头时重复计算另一个：

```powershell
python -m prospective.animation.render `
  --run-dir runs/animation_demo/<run_id> `
  --config configs/animation/mechanism_long.yaml `
  --clip neural

python -m prospective.animation.render `
  --run-dir runs/animation_demo/<run_id> `
  --config configs/animation/mechanism_long.yaml `
  --clip learning

# 全训练期总览：0 s 到训练结束，帧数与 FPS 由 config 决定
python -m prospective.animation.render `
  --run-dir runs/animation_demo/<run_id> `
  --config configs/animation/global_training_fast.yaml `
  --clip global
```

若希望一次渲染三个片段，使用 `--clip all`；保留的 `--clip both` 仍只渲染原有的 neural 与 learning 两个片段。

输出：

```text
<run_dir>/animations/feedforward_mechanism/
  neural_dynamics.mp4 或 neural_dynamics.gif
  learning_evolution.mp4 或 learning_evolution.gif
  global_training_dynamics.mp4 或 global_training_dynamics.gif
  manifest.json
```

- `neural_dynamics`：默认取训练末尾约 20 s 的连续窗口，展示多个 tutor passes 中的快速 `R -> U/r` 与慢 `V` 动力学。
- `learning_evolution`：按 tutor cycle 压缩的慢学习 montage。
- `global_training_dynamics`：从真实初始状态 `t=0` 均匀采样至训练末期。当前 `global_training_fast.yaml` 使用 1440 帧、10 FPS，对应约 144 s 视频，便于慢速观察 `U/V` bump delay。
- 播放速度由渲染配置中的 `animation.fps` 调节，不需要重新训练；`global_frame_count` 则控制从已有训练 history 中抽取多少个科学帧。降低 FPS 只延长每帧停留时间，不会增加或删除训练样本。
- `animation.render_progress: true` 会显示实际编码进度，例如 `302/1440 [00:39<02:47, 6.8 frame/s]`。进度来自 Matplotlib 的逐帧写入回调，包含完成帧数、帧速率和预计剩余时间；批处理时可设为 `false`。
- 上方三条同步曲线分别是连续 `R(x,t)`、按最终 learned preferred position 排列的 `U(x,t)` 和 `V(x,t)` profile。
- 若多个 competitive neurons 学到同一位置，profile 在该位置取群体均值；这只去除重复横坐标，不做空间平滑。下方节点仍逐个显示全部神经元。
- `U/V shape center` 使用减去各自空间最小值后的 center of mass，减少均匀 common-mode baseline 对 lag 读数的污染；原始 `U/V` 曲线和节点颜色不做基线扣除。
- 节点颜色分别表示 `R`、`U`、`V`；`r` 用 `U` 节点外的青色 halo 表示。
- 基础连线颜色表示当前 `J`；叠加颜色表示真实、有符号的 `delta_J`。
- 只显示 `abs(delta_J)` 最大的 top-k 连接，完整权重始终保留在独立的动态 matrix panel 中。
- 底部使用互不重叠的三栏布局：左侧节点/连接图、中间 `J(t)` 热图、右侧 `R/U/V` 色条。热图不再作为 inset 覆盖节点或连线。
- 全程片仅为热图显示把 postsynaptic rows 按最终 learned position 排序；训练、保存矩阵和连线索引仍保持原始 `J[post, pre]` 顺序。
- `manifest.json` 保存 source run、精确帧 index、播放速度、连接筛选规则和插值约定。

只要在当前 Conda 环境中可调用 `ffmpeg`，默认就输出 MP4；否则自动使用 Pillow 输出 GIF。视频编码器缺失不会影响训练。

### 5. Beta width-law sweep

该实验检验 Eq. 13：

```powershell
python -m prospective.experiments.run_beta_sweep `
  --config configs/analysis/beta_sweep.yaml
```

默认运行四个 `beta`、每个三个 seeds。输出包含每个 source run、`beta_sweep.csv` 和 `beta_width_test.png`。这比单次 toy run 更耗时，但仍使用 CPU。

## 配置文件

| 配置 | 用途 | 是否 paper-exact |
| --- | --- | --- |
| `fixed_theory_dynamics.yaml` | M1 固定理论 `J` 动力学 | 否，较小神经元数 |
| `feedforward_toy.yaml` | M2 主 toy 训练 | 否，学习率加快但保持时间尺度分离 |
| `animation_demo.yaml` | 16–32 神经元教学动画 | 否，visualization-scale |
| `global_training_fast.yaml` | 已有 animation run 的可调 FPS / frame-count 渲染参数 | 仅影响离线渲染，不改变模型 |
| `paper_reference_feedforward.yaml` | Figure 1 代表参数 | 参数参考；运行代价较高，不能保证完整训练时长与作者代码一致 |
| `prospective_shift.yaml` | M3 `(m, v)` frozen-weight grid | 读取 source run 参数 |
| `beta_sweep.yaml` | 多 seed 的 Eq. 13 width law | toy-scale quantitative check |

论文 Figure 1 报告的代表参数包括：`N=512`、`dt=0.005 s`、`tau_u=0.015 s`、`tau_v=0.6 s`、`m=0.2`、`A_R=30`、`sigma_R=5`、`eta_J=5e-4 Hz`、`alpha_J=1`、`v=26.8`。`paper_reference_feedforward.yaml` 保留这些数值；不要把较快的 toy 配置描述为逐参数精确复现。

## 每一步的更新顺序

模型在 `models/feedforward_toy.py` 中固定使用：

```text
t <- t + dt
-> compute tutor center and R(t)
-> I = J @ R
-> simultaneous Euler update of U and V using old U,V
-> r = relu(U_new)^2 / (1 + k * sum(relu(U_new)^2))
-> delta_J = dt * eta * r[:,None] * (R[None,:] - alpha * J**beta)
-> optional nonnegative clipping
```

矩阵约定始终是：

```text
J[i, j] = input neuron j -> competitive neuron i
J.shape = (n_competitive, n_input)
```

均匀网格上 `rho * dx = 1`，所以论文中的 `rho * integral` 离散成直接求和/矩阵乘法，代码不会重复乘 `rho` 或 `dx`。

### 前馈与递归训练的时间关系（M4 设计约定）

论文的讲解顺序是“先解析前馈 `J`，再加入递归 `W`”，但含递归的主实验并非严格串行预训练。论文描述的是 tutor 持续存在时，`J` 和 `W` 从初始状态**同步自组织**。一个离散步的目标顺序将是：

```text
compute tutor R(t)
-> I_ff = J @ R
-> I_rec = W @ r
-> update U,V and calculate r
-> update J from local pair (r_i, R_j)
-> update W from local pair (r_i, r_j)
```

`J` 与 `W` 的 plasticity update 属于同一个慢时间步，没有“本步先改完 J、再让 W 使用新 J”的因果优先级；两者都根据该步共同的神经活动快照计算，再同时提交。论文参数上通常要求 `alpha_W` 不小于 `alpha_J`，使早期 recurrence 不会压过 tutor 驱动并劫持空间表征。

M4 将默认提供两个明确区分的 protocol：

- `joint_from_random`：论文主结论对应的联合训练，`J`、`W` 从随机初值同步学习。
- `ff_pretrain_then_joint`：工程诊断/稳定性对照，先预训练 `J`，再开启 `W` 并继续联合学习；不得把它标成论文的主要自组织证据。

后续 recurrent 动画将沿用三种时间尺度，并同步显示 `J(t)`、`W(t)` 热图、`J @ R` 与 `W @ r` 的输入贡献。撤去 tutor 的测试只在联合训练结束、冻结 `J/W` 后进行，属于 M5 inference，不是 M4 的训练子阶段。

## Run 目录

```text
runs/<experiment>/<timestamp_seed>/
  config_resolved.yaml
  metadata.json
  status.json
  metrics.json
  training_history.npz
  weight_history.npz
  final_state.npz
  final_weights.npz
  figures/
    activity/
    connectivity/
    learning/
  prospective_probe/                 # M3 后
  animations/feedforward_mechanism/  # animation run 后
```

`metadata.json` 记录 Python、NumPy、git commit、dirty-worktree 状态、boundary mode 和 `J[post, pre]` 约定。`status.json` 区分 `running/completed/failed`；部分运行不会被静默标成成功。

## 核心指标如何阅读

- `gaussian_correlation`：对齐后的平均权重 profile 与理论 Gaussian shape 的相关。
- `learned_width`：自由 Gaussian fit 的诊断宽度。
- `theoretical_width`：Eq. 13 的无拟合预测。
- `width_relative_error`：两者相对误差。
- `translation_invariance_error`：对齐并归一化后不同 weight rows 的离散程度。
- `selective_row_fraction`：同时具有足够 row contrast 和超过初始化尺度的 competitive neurons 比例。
- `learned_position_coverage_fraction`：选择性 rows 的 peak 覆盖了多少 input position bins；它会暴露“少数赢家学得很好、其余神经元未被招募”的失败模式。
- `weight_update_norm`：最后一步的局部更新范数；它降低是收敛证据之一，不是唯一证据。
- `rate_vector_strength`：活动 bump 的圆形集中度，仅作 bump-validity gate。
- `rate_peak_to_baseline`：防止均匀活动被误解码为有效 bump。

在 `paper_reset` 分析中，靠近两端 `2.5 * sigma_R` 的 learned rows 不进入 Gaussian width 主统计，因为论文的解析推导假设 `L >> sigma_R` 并以实线积分代替有限区间积分。原始 rows 仍完整保存。

不能只凭权重热图出现斜带宣布成功。至少应联合检查：Gaussian correlation、width law、row variability、update norm、bump validity、多个 seeds，以及 `learning_enabled=false` / `m=0` controls。

## 方程到代码

| 论文对象 | 实现 |
| --- | --- |
| Eq. 1 moving Gaussian tutor | `stimuli/moving_tutor.py` |
| Eq. 2–3 `U,V` dynamics | `dynamics/competitive.py` |
| Eq. 5 quadratic rate / global inhibition | `dynamics/activation.py` |
| Eq. 6 local feedforward rule | `plasticity/feedforward.py` |
| Eq. 11–13 equilibrium Gaussian | `theory/equilibrium.py` |
| Eq. 18–22 first-Hermite shift | `theory/hermite.py` |
| M2 organization metrics | `analysis/weights.py` |
| M3 frozen-weight grid | `experiments/run_prospective_probe.py` |
| mechanism animation | `animation/` |

## 当前边界

- 已实现：M0 工程/解析基准、M1 固定理论权重、M2 前馈学习、M3 prospective probes、静态图，以及 neural / learning / global-training 三时间尺度动画。
- 未实现：recurrent `W` learning、自主 moving bump、uniform speed-current path integration、多层 feedforward hierarchy。
- `periodic_ring` 是明确标注的数值控制；论文默认条件仍为 `paper_reset`。
- Gaussian fit 只用于分析，不反馈进模型。
- 不使用反向传播、global loss、post-hoc symmetry projection 或权重平滑。

详细验收设计见 [.todo/TODO.md](.todo/TODO.md)。
