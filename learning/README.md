# Vafidis-style predictive plasticity toy model

本项目研究局部 predictive plasticity 能否在头方向（HD）网络中自组织出：

- 稳定且可移动的 bump；
- 由角速度驱动的路径积分；
- 可解释的 `HD→HD` 与 `HR→HD` 权重结构；
- 接近连续环吸引子的慢流形，而不只是若干离散吸引子。

训练只使用局部神经活动与预测误差信号，不使用 backpropagation、全局 loss 或 supervised RNN trainer。测试阶段冻结权重；除初始化 bump 外，darkness 测试不提供视觉输入或角速度输入。

> 速度增益或短时路径积分表现良好，并不足以证明形成了连续吸引子。本项目把 endpoint map、经验相位流、慢点、局部 Jacobian 与时间尺度分离作为主要的训练后诊断。

以下命令均从 `learning/` 目录运行。

## 快速开始

### 安装

```bash
conda activate random
python -m pip install -e .
```

若需要从 `environment.yml` 新建独立环境：

```bash
conda env create -f environment.yml
conda activate learning
python -m pip install -e .
```

### 训练单个网络

```bash
python -m learning.experiments.run_vafidis_toy \
  --config configs/experiments/vafidis_toy.yaml
```

Windows `cmd` 可写成单行：

```bat
python -m learning.experiments.run_vafidis_toy --config configs/experiments/vafidis_toy.yaml
```

添加 `--no-figures` 可只训练和测试而暂不绘图。每次运行都会保存解析后的完整配置、训练状态、指标和图像。

### 用同一基准配置派生不同运行

不需要再为每个训练预算复制一份完整 experiment YAML。训练 CLI 支持两种有序覆盖：

- `--profile PATH`：叠加一个可复用的局部 YAML；可以重复指定；
- `--set PATH=VALUE`：临时覆盖一个字段；可以重复指定，值按 YAML 解析。

合并优先级固定为：基准 experiment → diagnostics → `--profile`（从左到右）→ `--set`（从左到右）。字段名拼错或值类型不符会立即报错；最终完整配置仍保存为 run 目录中的 `config_resolved.yaml`。

例如，从正式 Vafidis baseline 派生一个 160,000 s 长训练：

```bash
python -m learning.experiments.run_vafidis_toy \
  --config configs/experiments/vafidis_toy.yaml \
  --set experiment_name=vafidis_release_ultralong \
  --set paths.runs_root=runs/vafidis_release_ultralong \
  --set simulation.train_duration=160000.0 \
  --run-id release_ultralong_seed42 \
  --no-figures
```

代码库提供了一个跨实验复用的短测试 profile：

```bash
python -m learning.experiments.run_vafidis_toy \
  --config configs/experiments/vafidis_toy.yaml \
  --profile configs/profiles/code_smoke.yaml \
  --set simulation.seed=43 \
  --run-id smoke_current \
  --no-figures
```

`code_smoke.yaml` 只保存相对于基准的运行预算差异：HD/HR 各 12 个单元、0.05 s 训练和短测试参数；实际启用哪些诊断仍只由 `vafidis_diagnostics.yaml` 控制。它保留 `dt=0.5 ms` 和当前 Vafidis 方程，因此适合检查代码路径是否完整，但结果没有科学解释意义。

运行前可只解析、校验并查看最终配置，不创建 run：

```bash
python -m learning.experiments.run_vafidis_toy \
  --config configs/experiments/vafidis_toy.yaml \
  --profile configs/profiles/code_smoke.yaml \
  --set "tests.gain_velocities=[-0.5, 0.0, 0.5]" \
  --print-config
```

`true`、`false`、`null`、数字和列表均按 YAML 值解析。包含空格或方括号时建议给整个 `PATH=VALUE` 加引号。`test_vafidis_toy --config ...` 入口也支持相同的 `--diagnostics-config`、`--profile` 和 `--set` 组合。

### 同时运行长训练与短测试

单条时间推进循环是有序的，不能把一个网络的 timestep 并行拆到多个 CPU core；更可靠的多核利用方式是启动彼此独立的进程。对于 Ryzen 7 9800X3D，可在两个已经激活同一 conda 环境的 PowerShell 终端中分别运行：

终端 A（长训练）：

```powershell
$env:OPENBLAS_NUM_THREADS = "6"
$env:OMP_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
python -m learning.experiments.run_vafidis_toy `
  --config configs/experiments/vafidis_toy.yaml `
  --set experiment_name=vafidis_release_ultralong `
  --set paths.runs_root=runs/vafidis_release_ultralong `
  --set simulation.train_duration=160000.0 `
  --run-id release_ultralong_seed42 `
  --no-figures
```

终端 B（代码 smoke）：

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m learning.experiments.run_vafidis_toy `
  --config configs/experiments/vafidis_toy.yaml `
  --profile configs/profiles/code_smoke.yaml `
  --set simulation.seed=43 `
  --run-id smoke_current `
  --no-figures
```

两个进程拥有独立 RNG、状态和输出目录。线程数只是 BLAS/OMP 上限；`N=60` 的小矩阵未必会从多线程 BLAS 获益，建议根据实测吞吐调整长任务为 4–7 threads，并给 smoke 与系统至少各保留一个硬件线程。不要让两个进程写入相同的显式 `--run-id`。

### 重测已有网络

无需重新训练即可改变诊断方案：

```bash
python -m learning.experiments.test_vafidis_toy \
  --run-dir /learning/runs/vafidis_release_parameter_baseline/<run_id> \
  --diagnostics-config configs/diagnostics/vafidis_diagnostics.yaml
```

不再使用命令行的单项 `--enable-*` 开关；诊断类别及其采样参数都只在上述同一份 hyper config 中修改。

### 重新分析或绘图

```bash
python -m learning.analysis.make_vafidis_figures --run-dir runs/<experiment>/<run_id>
python scripts/inspect_run.py runs/<experiment>/<run_id>
```

只重画 velocity phase-flow 图：

```bash
python -m learning.analysis.make_vafidis_figures \
  --run-dir runs/<experiment>/<run_id> \
  --phase-flow-only
```

### 测试代码

```bash
python -m pytest -q
```

若 Windows 临时目录清理失败，可使用项目内临时目录：

```bash
python -m pytest -q --basetemp runs/.pytest_tmp
```

## 配置体系

单鼠实验使用三类 YAML：

- `configs/experiments/`：模型规模、动力学、学习规则、训练刺激和训练时长；
- `configs/diagnostics/vafidis_diagnostics.yaml`：冻结权重后的唯一 hyper config，集中保存全部分组开关、测试时长和采样密度；
- `configs/profiles/`：可跨 experiment 复用的局部运行预算/功能测试覆盖，不是一份完整实验定义。

experiment YAML 不再嵌入或继承诊断配置。诊断入口显式接收唯一的 `vafidis_diagnostics.yaml`；该文件只能覆盖 `diagnostics`、`tests` 和测试相关 simulation 参数，不能改变模型、学习规则或训练过程。训练配置保存在 `config_resolved.yaml`，重测时实际使用的合并结果保存在 `test_config_resolved.yaml`。

### 单鼠实验

| 配置 | 当前规模 | 递归输入归一化 | 视觉 teacher |
| --- | ---: | --- | --- |
| `vafidis_toy.yaml` | HD 60 / HR 60 | `raw_sum` | von Mises |
| `vafidis_release_dt1ms_pilot.yaml` | HD 60 / HR 60 | `raw_sum` | von Mises |
| `vafidis_mammalian_heterogeneous.yaml` | HD 120 / HR 120 | `raw_sum` | heterogeneous GP |
| `vafidis_population_mean_heterogeneous.yaml` | HD 360 / HR 360 | `presynaptic_population_mean` | heterogeneous GP |
| `vafidis_population_mean_von_mises.yaml` | HD 360 / HR 360 | `presynaptic_population_mean` | von Mises |

文件名或历史 run 名中的 `n360` 不保证等于当前 YAML 的实际规模；应以 `config_resolved.yaml` 中的 `model.n_theta` 和 `model.n_hr` 为准。

`vafidis_toy.yaml` 现作为 release-parameter training baseline，而不再是加速调参版。它采用论文 Table 1 / 发布代码中的 `N=60`、`dt=0.5 ms`、80,000 s OU 训练、65 ms synaptic time constant、随机高斯初始权重、共同的局部学习率和无 clipping 权重更新。为消除 Eq. (4) 的局部 Euler stiffness，当前训练配置默认使用 `simulation.proximal_integration_method: exact_linear`；模型参数和全局固定时钟仍保持不变。当前所有正式 experiment YAML 都显式设置 `model.activation.max_rate: 0.15`，因此 `r_hd`、`r_hr`、PSP 与 prediction error 在内存中直接使用 release 的 kHz（数值等同 `ms^-1`）尺度，不再归一化到峰值 1。

proximal voltage 有两种可配置更新方法：

```yaml
simulation:
  dt: 0.0005
  proximal_integration_method: exact_linear  # 默认训练方法
  # proximal_integration_method: forward_euler  # 论文/release 数值对照
```

`exact_linear` 在每个 ordered timestep 内把刚更新的 distal voltage 和本步 proximal current 视为分段常值，并精确推进线性 Eq. (4) 子问题；它不是整个耦合网络的全局解析解。`forward_euler` 完全保留发布代码的 proximal 离散化，并继续执行 `dt*(gL+gD)/C < 2` 稳定性检查。历史 `config_resolved.yaml` 若没有该字段，加载时回退为 `forward_euler`，从而保持旧 run 的重测语义。也可临时使用 `--set simulation.proximal_integration_method=forward_euler` 切换。

release baseline 的相应数值为：

- 固定 `HD→HR` 权重：`2 / 0.15 = 13.333333...`；
- 随机可塑权重标准差：`1 / sqrt(60 + 60) = 0.091287...`；
- 秒单位训练循环中的学习系数：`0.05 × 1000 = 50`。这里 `1000` 只换算 `dt_ms` 与 `dt_s`，不再包含任何 `f_max³` firing-rate 换算。

当前 baseline 已按论文 Eq. (2)–(4) 动态推进 distal current、distal voltage 与 proximal voltage，并像 release 一样学习包含 diagonal 的完整 `HD→HD` 矩阵。它与 release 的剩余区别包括默认 exact-linear proximal 子步、秒单位、可复现 RNG，以及按论文修正 `fly_rec.py` 中 HD→HR 低通更新漏写 `dt` 的歧义。历史 `config_resolved.yaml` 若没有 `activation.max_rate`，加载时仍保留旧的 unit-peak 语义，因此旧权重可以原样重测；但旧的归一化权重不能与新的 `max_rate: 0.15` 配置交叉混用。

`visual.normalize_peak` 只控制 visual-current profile 的定义，不会归一化神经元 firing rate 或 prediction error。连接上的 `presynaptic_population_mean` 也是一个单独且显式的 scaling control；release baseline 与主 heterogeneous 配置使用 `raw_sum`。

`vafidis_release_dt1ms_pilot.yaml` 是研究时间步精度时留下的历史副产物，为旧命令和结果追溯而保留。新的长训练、短预算和功能测试应优先从 `vafidis_toy.yaml` 出发，使用 `--profile`/`--set` 派生，避免继续复制完整 experiment YAML。

### 诊断方案

所有冻结权重诊断只使用 `configs/diagnostics/vafidis_diagnostics.yaml`。其中九个稳定的功能组以布尔值独立开关：

```yaml
diagnostics:
  bump_maintenance: false
  path_integration_and_pi_error: false
  pva_spectrum_and_visualization: false
  velocity_gain: false
  trajectory_and_fixed_points: true
  weight_snapshots_and_development: false
  bump_diffusion: false
  timescale_separation: false
  velocity_dynamics_and_phase_flow: false
  reuse_cached_dependencies: true
```

| 分组 | 执行内容 |
| --- | --- |
| `bump_maintenance` | cue-off bump maintenance 与零速度漂移 |
| `path_integration_and_pi_error` | constant-velocity PI、OU PI、OU ensemble 与 PI error |
| `pva_spectrum_and_visualization` | HD tuning/PVA 可视化、Ramesan PCA spectrum 与 slow-manifold 分析 |
| `velocity_gain` | commanded/decoded velocity gain |
| `trajectory_and_fixed_points` | 零输入 trajectories、stable FP 与 unstable basin boundary |
| `weight_snapshots_and_development` | 权重结构、snapshot、norm 与发展过程 |
| `bump_diffusion` | 噪声下 bump ensemble diffusion |
| `timescale_separation` | normal/tangential timescale assay |
| `velocity_dynamics_and_phase_flow` | velocity trajectory sweep、phase flow、FP 与 basin |

各组在计算、保存和绘图阶段使用同一套开关。算法依赖不等于启用另一个诊断组：例如 trajectory 的 Clark overlap decoder 需要 HD tuning template，代码会优先复用当前 run 的 `hd_tuning_history.npz`，没有缓存时只计算并保存这一依赖，不会额外运行 PVA spectrum、PI 或 diffusion。PVA/slow-manifold 与 timescale 需要 trajectory source 时采用相同规则。

从 `learning/` 目录运行当前配置（默认仅开启 trajectory/FP 组）：

```bash
python -m learning.experiments.test_vafidis_toy \
  --run-dir runs/vafidis_release_parameter_baseline/<run_id> \
  --diagnostics-config configs/diagnostics/vafidis_diagnostics.yaml
```

从仓库根目录运行时，在路径前加 `learning/`：

```powershell
python -m learning.experiments.test_vafidis_toy `
  --run-dir learning\runs\vafidis_release_parameter_baseline\<run_id> `
  --diagnostics-config learning\configs\diagnostics\vafidis_diagnostics.yaml
```

需要横向比较实验时，使用 `configs/analysis/`：

- `visual_current_noise_std_comparison.yaml`：视觉电流 OU noise sweep；
- `noise_by_neuron_count_comparison.yaml`：noise × neuron-count 网格；
- `heterogeneous_30_mouse_hyper.yaml`：30 个独立 seed 的 Clark-style 群体分析。

运行方式：

```bash
python -m learning.experiments.run_attractor_robustness \
  --robustness-config configs/analysis/heterogeneous_30_mouse_hyper.yaml
```

该 30 鼠 preset 会覆盖基础 heterogeneous 配置，使每只模拟小鼠使用 HD 360 / HR 360；当前 seeds 为 42–71。

## 模型与缩放约定

### 成对 HD 几何

`N` 个 HD 单元表示 `N/2` 个唯一 heading，每个方向包含一对 HD 单元。调谐曲线统计覆盖全部 `N` 个单元；绘制 activity heatmap 和权重矩阵时按训练后 COM 排序，纵轴使用 neuron ID，而不是把不均匀 COM 强行放在均匀角坐标上。

### 训练与测试

训练期间，视觉 cue 充当局部学习所需的 teacher，并与 OU 角速度轨迹共同驱动网络。测试期间：

1. 加载训练后的完整状态；
2. 冻结所有可塑权重；
3. 根据测试条件提供 visual/velocity 输入；
4. 在 darkness 条件下令两者为零，仅用短暂 cue 初始化 bump。

因此，darkness 中的后续轨迹反映网络自身动力学，而不是视觉 cue 的持续钳制。

### Population scaling

两类递归电流约定可直接比较：

- `raw_sum`：保留原始 Vafidis toy model 的求和形式；
- `presynaptic_population_mean`：`HD→HD` 输入除以 `N_HD`，每个 `HR→HD` wing 除以 `N_HR/2`。

固定的一对一 `HD→HR` 通路和外部电流保持 `O(1)`，不参与 population-mean 除法。代码不人为强制 `HD→HD` 对称或左右 `HR→HD` 反对称；这些结构必须由 online local learning 自行形成。

## 视觉 teacher 与调谐曲线

### von Mises teacher

同质 cue 具有平移对称性，适合检查学习规则在理想输入下能否形成环状结构。

### Heterogeneous teacher

heterogeneous cue 由圆周上的 wrapped Gaussian process 生成，再经平滑正值变换。每条 profile 先归一到相同的角平均值，最后统一乘以共享幅值；这样保留形状、峰宽和多峰性的异质性，同时避免每个神经元具有任意不同的整体增益。

示例图默认以 `4 × 4` 展示 16 条 profile。COM-aligned 分析会先让每条训练后 tuning curve 除以自身最大值，再按各自 COM 平移；不会用全体神经元的全局最大值归一化。粗黑线表示全体神经元的均值，std curve 使用相同的对齐和归一化顺序。

### Clark-style 30 鼠分析

每只模拟小鼠独立初始化、独立采样训练轨迹和 heterogeneous profiles。跨鼠图中：

- 浅灰线：单鼠内先对神经元求平均的 COM-aligned mean/std；
- 粗黑线：再对小鼠等权平均；
- Figure 4A：各鼠 COM，并用 Kuiper test 检验圆周均匀性，再用 Benjamini–Hochberg 校正多重比较；
- Figure 4B：神经元数量匹配的均匀圆周 null；
- Figure 4C：不同 `N_sub` 下的 tuning correlation matrix。

Figure 4C 使用未中心化二点函数

\[
C = \frac{R^\mathsf{T}R}{N_{\mathrm{sub}}},
\]

其中 `R` 是选中神经元的调谐响应。`epsilon_circ` 是该矩阵与其最佳 circulant 近似之间的相对 Frobenius 误差；越接近 0，circulant 特征越强。heatmap 的角度轴是 stimulus bin，不意味着各神经元 COM 必须均匀分布。

## 训练后诊断

| 诊断 | 主要问题 | 典型输出 |
| --- | --- | --- |
| HD tuning / COM | 是否形成一致、尖锐且覆盖圆周的 tuning | tuning、COM heatmap、mean/std |
| PI / OU / gain | bump 能否随角速度移动 | tracking error、gain curve |
| endpoint map | darkness 中是连续环还是离散 basin | trajectory、initial→final map |
| slow manifold | 是否存在覆盖圆周的低速一维流形 | PCA、`q`、tangent flow、Jacobian |
| timescale separation | 法向收缩是否远快于切向漂移 | parallel/perpendicular timescale |
| velocity phase flow | 低速输入下是否 depin 或 stick–slip | trajectories、`F_v(θ)`、FP rings |
| noise / diffusion | 噪声下是局部扩散、跳 basin 还是回缩 | MSD、endpoint distribution |

### Endpoint map 与三种 decoder

初始 cue 均匀铺在 `[-π, π)`，随后在无视觉、无速度输入下弛豫。支持三种读出：

- PVA：默认主读出；
- peak：最大 firing-rate 单元；
- Clark overlap：与参考 tuning template 的最大 overlap。

理想连续环应呈近水平 trajectory 和近恒等的 initial→final map；多条初始状态收缩到少数水平行，则说明存在离散 attractor basins。

endpoint 图中的青色水平窄条标记终点平台给出的 stable fixed point；若整体 endpoint map 已有充分的平台收缩证据，也保留恰好落在 fixed point 上的 stationary singleton。橙色窄条标记相邻 cue 落入不同 basin 时推断出的 unstable boundary：只有相邻位移直接形成负到正的 bracket 时才插值，否则取该采样区间的中点；不会跳过近零位移 probe。条带宽度仅用于可见性，不表示 basin 或 governed region 的大小。

### Slow-manifold 与局部 Jacobian

自主动力学使用完整状态

\[
x=[r_{HD\to HR}^{LP},\ r_{HR}^{prev},\ i_{HD}^{distal},\ v_{HD}^{distal},\ v_{HD}^{proximal}],
\qquad
f(x)=\frac{G_{\Delta t}(x)-x}{\Delta t},
\]

并定义 Ramesan-style 慢点指标

\[
q(x)=\frac{1}{2}\lVert f(x)\rVert_2^2.
\]

分析流程为：密集初始化并在 darkness 中采样状态，先确认角度覆盖，再拟合周期一维流形；将经验流投影为切向速度，并在完整状态上计算局部 Jacobian。PCA 只负责前三主成分的可视化，不用于替代动力学状态。`ramesan_pca_variance_rank.png` 分别报告 HD+HR firing-rate state 与 PVA 实际依赖的 paired-HD angular-rate statistic 的完整 explained-variance/rank 谱；若前三成分累计比例较低，三维图只能作定性展示。

近似连续环至少应同时满足：

1. 慢状态覆盖并闭合成环，而不是少量孤立点；
2. 整个环上的切向流都较小；
3. 切向存在近中性的局部模态且与流形切线对齐；
4. 法向模态稳定，并与切向形成谱隙；
5. 法向恢复显著快于切向漂移；
6. endpoint 不坍缩到少数 basin。

单独的低 `q` 也可能只是普通稳定 fixed point。权重矩阵的特征值不是动力学 Jacobian 的特征值，不能据此宣称存在零模态。

### 时间尺度分离

扰动被拆成流形切向和平面法向分量。当前保守判据为

\[
\frac{T_{\parallel}^{p10}}{T_{\perp}^{p90}} \ge 10.
\]

它检验“较快的切向漂移时间”是否仍至少是“较慢的法向恢复时间”的 10 倍。该指标是有限观测窗内的操作性检验，不等价于严格证明无限时间连续吸引子。

### Velocity trajectory 与经验相位流

所有轨迹从同一 bump 状态出发，在多个恒定 `v` 下运行约 30 s。连续吸引子应产生近直线的 unwrapped angle；离散景观通常表现为停滞、加速和跨越障碍组成的 stick–slip 波形。

dense probes 直接从网络轨迹估计

\[
F_v(\theta) \approx \dot{\theta}.
\]

`F_v` 从正变负的位置是经验稳定 FP，从负变正的位置是经验不稳定 FP；若全圆同号，则该速度下没有真实 FP，系统已经 depin。不同初态即使接受相同速度，也可能因内部高维状态不同而具有不同的瞬时切向驱动力，因此不能只用一维静态势垒解释所有轨迹。

### Diffusion 的解释边界

若网络由离散 basin 构成，噪声下的“扩散曲线”可能混合 basin 内回缩、偶发跨 basin 跳跃与真正的环向 diffusion。应联合查看 endpoint map、条件 MSD 和跳跃分布；不要把单条总体 MSD 曲线作为连续吸引子的证据。

## 输出结构

单次运行默认写入 `runs/<experiment>/<run_id>/`：

```text
config_resolved.yaml          训练与默认诊断的完整配置
test_config_resolved.yaml     最近一次独立重测配置
trained_weights.npz           训练后权重、偏好角和视觉 profiles
training_history.npz          训练轨迹摘要
weight_history.npz            权重快照
hd_tuning_history.npz         训练后调谐曲线
bump_attractor_trajectory_history.npz
slow_manifold_diagnostics.npz
timescale_separation_history.npz
velocity_trajectory_sweep_history.npz
test_metrics.json             常规测试指标
figures/                      常规训练和调谐图
figures/diagnostics/          endpoint、慢流形和相位流图
```

并非每个诊断都会启用，因此部分文件可能不存在。hyper experiment 的汇总结果保存在 `reports/attractor_robustness/<preset>/<run_id>/`。

## 代码结构

```text
src/learning/
  analysis/       指标、slow manifold、phase flow 与综合绘图
  common/         圆周角度、数组和随机数工具
  config/         YAML 加载、继承与 schema 校验
  connectivity/   权重初始化和结构约束
  dynamics/       HD/HR 更新、激活函数与自主映射
  experiments/    训练、重测、slow-manifold 和 hyper 入口
  io/             run 目录与状态存取
  models/         Vafidis toy state 及单步组合
  plasticity/     局部 predictive learning 与 traces
  plotting/       activity、heading、weights 和 diagnostics 图
  stimuli/        velocity、von Mises 与 heterogeneous cue
tests/             单元测试和回归测试
notebooks/         文献公式与诊断方法说明
configs/           experiment、diagnostics 和 analysis 配置
```

理论与方法笔记见 `notebooks/Vafidis.ipynb`、`notebooks/sagodi.ipynb` 和 `notebooks/ramesan.ipynb`。未完成事项及实验决策记录见 `.todo/TODO.md`。
