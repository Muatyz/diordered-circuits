# Emina & Kropff toy model：复现目标与代码架构

## 1. 总目标

构建一个规模较小、机制可逐项检查的 Emina & Kropff (2026) toy model，用数值实验回答以下问题：

1. 平移不变的单向移动 Gaussian tutor，能否通过论文的局部 Hebbian + weight-dependent decay 规则，把随机非负前馈权重 \(J\) 整理成近似 Gaussian、只依赖相对位置的连接？
2. 学得的连接宽度是否随 decay exponent \(\beta\) 遵循论文的 equilibrium prediction？
3. firing-rate adaptation 是否在不引入非对称递归权重的情况下产生 prospective shift？
4. 在前馈结果成立之后，论文的 recurrent learning rule 能否学出支持 autonomous moving bump 的连接？
5. 最后再检验 uniform speed current 是否能在论文的一阶近似适用范围内调节 bump 速度，形成一维、单向路径积分。

本项目首先复现“机制链”，不以逐像素重画论文 Figure 为目标：

```text
moving tutor statistics
    -> local feedforward learning
    -> Gaussian / translation-invariant J
    -> adaptation-induced prospective shift
    -> recurrent learning
    -> self-sustained moving bump
    -> speed-current modulation
    -> unidirectional path integration
```

## 2. 论文忠实性与明确边界

### 2.1 必须保留的原文机制

- 输入层活动：移动的 normalized Gaussian profile，且 \(A_R\) 表示 integrated drive，不误作 peak height。
- competitive layer：

  \[
  \tau\dot U=-U-V+I,
  \qquad
  \tau_v\dot V=-V+mU.
  \]

- firing rate 与 global divisive inhibition：

  \[
  r_i=\frac{[U_i]_+^2}{1+k\sum_j[U_j]_+^2}.
  \]

- 前馈局部学习规则：

  \[
  \dot J_{ij}
  =\eta_J r_i\left(R_j-\alpha_JJ_{ij}^{\beta}\right).
  \]

- recurrent 阶段使用论文对应的局部规则，而不是复制、平移或手工对称化前馈权重。
- 权重保持非负；若按论文 simulation 约定，数值更新后执行 nonnegative clipping。
- neural dynamics 与 learning dynamics 使用明确的时间尺度分离，并分别记录时间步长与更新顺序。
- prospective coding 来自 adaptation dynamics，不得向权重中手工加入前向偏移。

### 2.2 默认不允许的“帮助模型成功”操作

- 不使用 backpropagation、autograd、global loss 或 supervised fitting 来更新突触。
- 不对学得的权重做 post-hoc circulant averaging、Gaussian smoothing、左右镜像或强制对称化。
- 不根据目标 bump 位置修正网络状态或选择性重置失败神经元。
- 不以 decoder calibration 掩盖动力学 gain error。
- 不因 toy run 较慢而擅自改变论文方程；任何近似或加速必须有独立开关、测试和标注。
- Gaussian fitting 只用于分析学得结构，拟合结果不得反馈进模型。

### 2.3 必须明确标注的 toy-model 简化

- 首个版本为一维有限神经元离散模型，不宣称是完整 entorhinal circuit。
- 论文理论采用长线段、\(L\gg\sigma_R\)、到边界后 reset 的处理；默认实验应保留 `paper_reset`。可增加 `periodic_ring` 作为数值控制，但图和指标必须与原文条件分开报告。
- 首个里程碑只含 feedforward connectivity。没有 recurrent self-sustenance 时，不把活动称作 continuous attractor。
- 论文最后实现的是 unidirectional path integration，不能描述为完整的 signed HD ring integrator。
- 若 toy 参数与论文表格参数不同，配置名和 README 必须明确使用 `toy`，不能称为 paper-exact reproduction。

## 3. 数学约定

全项目固定以下矩阵方向，避免原文连续记号中 \(x,x'\) 的歧义：

- `J[i, j]`：input neuron `j` \(\to\) competitive neuron `i`。
- `W[i, j]`：competitive neuron `j` \(\to\) competitive neuron `i`。
- `R.shape == (n_input,)`。
- `U.shape == V.shape == r.shape == (n_competitive,)`。
- `J.shape == (n_competitive, n_input)`。
- `W.shape == (n_competitive, n_competitive)`。

连续积分离散化必须写清 normalization convention。若均匀网格上

\[
\rho\Delta x=\frac{N}{L}\frac{L}{N}=1,
\]

代码可直接使用矩阵乘法，但测试应验证没有重复乘或漏乘 \(\rho\)、\(\Delta x\)。角度/位置 decoder 在 periodic control 中必须使用 circular population vector，而不是普通质心。

## 4. 实施范围与分阶段关卡

每个阶段只有通过自己的验收条件后才能进入下一阶段。首轮批准后建议只实施 M0--M3。

### M0：最小工程骨架与 analytic oracle

- [ ] 创建 package、配置加载、run-directory、测试和 README 骨架。
- [ ] 实现 normalized Gaussian、离散积分、位置距离和固定随机数生成器。
- [ ] 实现论文 equilibrium width prediction：

  \[
  \sigma_J=\sqrt{\frac{3\beta}{2-\beta}}\sigma_R,
  \qquad
  \sigma_u=\sqrt{\frac{2\beta+2}{2-\beta}}\sigma_R.
  \]

- [ ] 对 \(\beta\le0\) 或 \(\beta\ge2\) 明确报错，不让无效理论参数静默运行。
- [ ] 保存 resolved config、随机 seed、代码版本信息和运行状态。

验收：数学辅助函数、配置验证和 run I/O 测试通过；尚不要求学习结果。

### M1：固定理论前馈权重下的 neural dynamics

- [ ] 先使用解析 Gaussian `J`，不学习权重。
- [ ] 验证移动 tutor 能驱动稳定、局域的 \(U,V,r\) bump。
- [ ] 比较 `m=0` 与 `m>0` 时 adaptation lag、bump width 和 center offset。
- [ ] 做 `dt` convergence 检查，排除 Euler 离散产生的虚假前移。
- [ ] 可视化 tutor、\(U\)、\(V\)、\(r\) 的时空热图和代表性切片。

验收：活动有限、非 NaN、bump 可解码；改变 `dt` 后主要指标稳定；前移方向不由 decoder 或边界伪影产生。

### M2：从随机权重学习 feedforward connectivity（核心 MVP）

- [ ] 从小的非负随机 `J` 开始；不使用理论 Gaussian 初始化作为主实验。
- [ ] 逐步更新 neural state 与论文的前馈 learning rule。
- [ ] 保存稀疏 weight checkpoints，不保存每一步完整矩阵。
- [ ] 比较随机初态、训练中期和 equilibrium 附近的 `J`。
- [ ] 将每一行按 postsynaptic preferred location 对齐，计算平均 relative-displacement profile。
- [ ] 用理论 \(\sigma_J\) 作无拟合参数的 overlay；另报告自由 Gaussian fit 仅作为偏差诊断。
- [ ] 运行至少多个 seeds，区分规则效果与偶然 realization。

主要验收指标：

- `weight_gaussian_correlation`：对齐平均 profile 与理论 Gaussian 的相关。
- `weight_width_relative_error`：学得宽度对理论 \(\sigma_J\) 的相对误差。
- `translation_invariance_error`：不同 postsynaptic rows 对齐后的离散程度。
- `weight_update_norm`：训练后期平均更新范数是否下降。
- `weight_norm`、`weight_max`：是否 runaway 或完全 collapse。
- bump validity：PVA/质心强度、peak-to-baseline contrast、有效宽度。

最小对照：

- `learning_enabled=false`：随机权重不应自行变成 Gaussian。
- `m=0`：检查 adaptation 对竞争和神经元覆盖的必要性。
- `decay_enabled=false`：预期权重失控；仅作短时机制对照并设置安全终止。
- 非均匀 tutor occupancy：检验连接结构是否继承经验偏置。

### M3：prospective coding

- [ ] 在 M2 学得的冻结 `J` 上测量 activity center 相对 tutor center 的 signed offset。
- [ ] 扫描 adaptation strength `m` 和 tutor velocity `v`。
- [ ] 报告空间偏移 \(\Delta z\) 与 anticipation time \(\Delta z/v\)，并避免在 \(v\approx0\) 时计算不稳定比值。
- [ ] 使用 `m=0`、`v=0` 和 reversed-motion 控制区分 adaptation、静态偏置与边界效应。
- [ ] 可选：实现 first-Hermite projection，比较 \(\Delta z\approx\sigma_u\gamma\)。
- [ ] 可选：串联少量冻结的 feedforward layers，检验 shift 是否在误差可控范围内近似随层数累积。

验收：prospective shift 的符号随运动方向反转；`m=0` 时显著减弱；结论跨多个 seeds 和较小 `dt` 保持。

### M4：recurrent learning（M0--M3 通过后再批准）

- [ ] 加入 `W @ r`，实现论文 recurrent learning rule。
- [ ] 主 protocol 使用 `joint_from_random`：tutor 持续存在，`J` 与 `W` 从随机初值同步学习；两种 plasticity rule 使用同一神经活动快照计算并在同一慢时间步提交，不能因代码书写顺序引入隐式先后依赖。
- [ ] 增加明确标为工程对照的 `ff_pretrain_then_joint`：先形成 `J`，再开启 `W` 并继续联合学习；不得用这一 warm-start 单独支持论文“同时自组织”的主张。
- [ ] 同时记录 `J`、`W`、`delta_J`、`delta_W` 及 `J @ R`、`W @ r`，检查 recurrence 是否在空间表征形成前压过 tutor；重点扫描论文提出的相对强度条件 `alpha_W >= alpha_J` 附近，而不是默认 recurrence 越强越好。
- [ ] 复现或简化论文的 `(alpha_J, alpha_W)` phase diagram。
- [ ] 比较学得 `W` 的 Gaussian profile、平移不变性、谱和稳定性。

验收：连接由局部规则学出；没有手工复制 `J` 或强制对称化；训练期间 bump 始终可辨认。

### M5：撤去 tutor 后的 autonomous moving bump

- [ ] 冻结权重并设 feedforward tutor input 为零。
- [ ] 分别测试 bump survival、静态稳定和 adaptation-driven traveling bump。
- [ ] 比较 intrinsic speed 与论文 projection prediction，并把论文的经验校正系数作为“论文报告值”，不得偷偷吸收到模型参数。
- [ ] 区分 bump collapse、stationary bump、coherent moving bump 和全局高活动四种状态。

验收：多个初始位置均能形成相似的局域 traveling bump；速度不是 transient release shift 的误判。

### M6：uniform speed current 与单向路径积分

- [ ] 实现 uniform `I_speed(t)`，保持位置 tutor 关闭。
- [ ] 先验证 baseline shift \(\to\) bump width \(\to\) bump speed 的中间链条。
- [ ] 只在小扰动区间拟合/验证一阶 speed gain。
- [ ] 使用 constant-speed、piecewise-speed 和平滑 time-varying speed protocol。
- [ ] 报告 position error、speed gain、bump validity 和超出线性范围后的失效。

验收：网络输出位置近似速度积分；结果明确标注为一维、单向 PI，不外推为 signed HD integration。

### M7：与 Clark/Vafidis 的桥接分析（可选研究扩展）

- [ ] 固定 mature weights 后计算 neural-state Jacobian，和 learning-operator stability 分开命名。
- [ ] 检查 transverse stability、tangent drift 和 Fourier spectral doublets。
- [ ] 加入有限尺寸、神经元异质性或 teacher heterogeneity，检验逐元素无序下统计平移对称性是否保留。
- [ ] 用相同训练轨迹比较 Emina Hebbian rule 与 Vafidis predictive local rule，但不得混合两者更新项。

## 5. 建议代码架构

```text
prospective/
  .todo/
    TODO.md
  README.md
  pyproject.toml
  configs/
    experiments/
      feedforward_toy.yaml
      animation_demo.yaml
      prospective_shift.yaml
      recurrent_toy.yaml              # M4 后再启用
      path_integration_toy.yaml        # M6 后再启用
    analysis/
      beta_sweep.yaml
      adaptation_velocity_sweep.yaml
  notebooks/
    Emina_Kropff.ipynb                 # 解释/探索，不承载正式模型逻辑
  src/
    prospective/
      __init__.py
      common/
        arrays.py
        geometry.py
        random.py
      config/
        load.py
        schema.py
      stimuli/
        moving_tutor.py
      dynamics/
        competitive.py
        activation.py
      plasticity/
        feedforward.py
        recurrent.py                   # M4
      theory/
        equilibrium.py
        hermite.py                     # M3，可选
      models/
        feedforward_toy.py
        recurrent_toy.py               # M4
      analysis/
        decoding.py
        weights.py
        metrics.py
      plotting/
        activity.py
        connectivity.py
        learning.py
        prospective.py
      animation/
        storyboard.py
        feedforward_scene.py
        render.py
      io/
        run_dir.py
        save_load.py
      experiments/
        run_feedforward.py
        run_prospective_probe.py
        run_recurrent.py                # M4
        run_path_integration.py         # M6
  tests/
    test_gaussian.py
    test_config.py
    test_shapes.py
    test_competitive_dynamics.py
    test_feedforward_learning.py
    test_equilibrium_theory.py
    test_decoding.py
    test_animation_data_mapping.py
    test_smoke_feedforward.py
  runs/                                # gitignored，原始运行产物
  reports/
    figures/                           # 经选择、可追溯的汇总图
    tables/                            # sweep 汇总 CSV
    notes/                             # 诊断记录与偏差说明
```

架构原则：

- `dynamics/` 只负责固定权重下的快速神经动力学。
- `plasticity/` 只负责给定 pre/post state 的局部权重更新。
- `theory/` 提供独立解析 prediction，不能调用 simulation 的拟合结果。
- `models/` 规定单步更新顺序并组合以上模块。
- `analysis/` 和 `plotting/` 不能修改模型状态。
- `animation/` 只从已保存的 state/weight history 构造科学叙事镜头，不重新执行或修改模型动力学。
- `experiments/` 只负责配置、循环、checkpoint 和调度，不重复数学公式。
- notebook 只读取 package API 与已保存 run；验证过的逻辑应下沉到 `src/`。
- 不创建按论文图号组织的大量孤立脚本；绘图函数按科学对象复用。

## 6. 配置设计

所有实验参数来自 YAML，并在运行开始时解析为带类型和范围校验的 dataclass。命令行只允许覆盖少量运行参数，最终值必须写回 `config_resolved.yaml`。

建议配置分组：

```yaml
experiment:
  name: feedforward_toy
  seed: 11
  output_root: runs

geometry:
  length: 100.0
  n_input: 128
  n_competitive: 128
  boundary_mode: paper_reset

tutor:
  speed: 5.0
  sigma: 5.0
  integrated_drive: 30.0

neural:
  dt: 0.005
  tau_u: 0.015
  tau_v: 0.6
  adaptation_strength: 0.2
  inhibition_strength: null  # 确认 paper/toy convention 后填写

feedforward_learning:
  enabled: true
  eta: 0.0005
  alpha: null               # 确认 normalization 后填写
  beta: 0.5
  nonnegative_clip: true

simulation:
  duration: null             # 通过短跑收敛诊断后确定
  warmup_duration: 0.0
  state_sample_interval_steps: 20
  weight_snapshot_interval_steps: 1000
  divergence_threshold: 1.0e6

analysis:
  transient_duration: null
  bump_strength_min: null
  fit_gaussian_for_diagnostics: true

animation:
  enabled: false
  fps: 30
  render_progress: true
  display_top_k_connections: 64
  update_highlight_mode: top_k_absolute_delta
  render_matrix_inset: true
  neural_window: null
  learning_frame_interval_cycles: 1
  output_format: mp4
```

注意：上面的数值只是 notebook 中建议的 toy 起点，不应在实现前被当成已经验证的 paper-exact 参数。`null` 参数必须在审阅论文的 normalization convention 后显式填写，配置加载器不应静默猜测。

建议 CLI：

```powershell
python -m prospective.experiments.run_feedforward --config configs/experiments/feedforward_toy.yaml
python -m prospective.experiments.run_prospective_probe --run-dir runs/feedforward_toy/<run_id>
python -m prospective.animation.render --run-dir runs/feedforward_toy/<run_id> --config configs/experiments/animation_demo.yaml
```

## 7. Run 目录与可复现性

每次运行创建独立目录：

```text
runs/<experiment_name>/<YYYYMMDD-HHMMSS>_seed<seed>/
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
    prospective/
  animations/
    feedforward_mechanism/
      manifest.json
      neural_dynamics.mp4
      learning_evolution.mp4
```

要求：

- `metadata.json` 记录 Python/package 版本、git commit、dirty-worktree 标志、开始/结束时间和矩阵 convention。
- `status.json` 明确 `running/completed/failed`，失败时保存原因；不把部分运行误当成功结果。
- 所有随机性使用显式 `numpy.random.Generator`，不依赖全局 RNG。
- history 文件记录真实采样时间和 step index，绘图不能猜采样间隔。
- checkpoint 间隔由配置控制；大数组不写 JSON。
- `runs/` 默认 gitignored；进入 `reports/` 的图必须能追溯到 source run 和 resolved config。
- 同一 resolved config 与 seed 应产生数值一致的 CPU 结果（容许平台浮点微差）。

## 8. 首轮必须生成的可视化

M0--M3 的最小图集：

1. `tutor_and_activity_heatmaps.png`
   - tutor \(R\)、膜电位 \(U\)、adaptation \(V\)、rate \(r\) 随位置和时间的热图。
2. `activity_profile_snapshots.png`
   - 同一时刻叠加 tutor、\(U\)、\(V\)、\(r\)，直接看 adaptation lag。
3. `feedforward_weights_initial_mid_final.png`
   - 随机初态、训练中期、最终 `J`，使用一致色标并注明矩阵方向。
4. `aligned_weight_profile_vs_theory.png`
   - rows 对齐后的均值与 seed/row 离散带；叠加理论 \(\sigma_J\) Gaussian。
5. `learning_diagnostics.png`
   - weight update norm、weight norm、Gaussian correlation、translation-invariance error 随训练时间。
6. `prospective_shift_vs_time.png`
   - tutor center、decoded activity center 和 signed shift；标出 reset/transient 区域。
7. `prospective_shift_parameter_map.png`
   - \((m,v)\) 上的 shift、bump validity 和失败 mask；无有效 bump 的格点不得绘成合法 shift。
8. `beta_width_test.png`
   - 多个 \(0<\beta<2\) 下学得 \(\sigma_J\) 与理论曲线，附多个 seeds 的不确定性。

### 8.1 机制解释动画：`feedforward_mechanism`

增加一个接近 3Blue1Brown 叙事风格、但严格由 simulation 数据驱动的分层动画，用于把移动 tutor、神经状态和局部突触更新放在同一个空间坐标系中解释。动画属于教学性可视化，不替代 M1--M3 的定量图和统计验收。

#### A. 画面中的科学对象

所有层共享同一个从左到右递增的位置坐标。神经元的水平位置由其 preferred position 决定，避免视觉布局暗示不存在的连接结构。

1. **Tutor trajectory / continuous stimulus band**
   - 最上方绘制连续 Gaussian curve：

     \[
     R(x,t)=A_R\mathcal N(x;z(t),\sigma_R),
     \qquad z(t)=vt.
     \]

   - 以移动竖线或标记显示真实 tutor center \(z(t)\)，同时显示当前 simulation time、position 和 velocity。
   - `paper_reset` 模式下，接近边界和 reset 的帧必须明确标记；不得用平滑跨界动画掩盖 reset discontinuity。

2. **Input-neuron band**
   - 第二行放置 \(N_{\mathrm{in}}\) 个离散 input neurons，节点颜色表示当前引导放电率 \(R_j(t)\)。
   - 节点顺序与 `x_input[j]` 一致；颜色条注明变量、单位或归一化约定，并在同一个 clip 内使用固定范围。
   - 连续 Gaussian 与离散 \(R_j\) 同屏显示，以呈现“连续 tutor profile 被神经元群体采样”。

3. **Feedforward-synapse band**
   - 从 input neuron \(j\) 到 competitive neuron \(i\) 的连线严格对应 `J[i, j]`。
   - 当前突触强度 \(J_{ij}(t)\) 使用非负 sequential colormap 编码；线宽或透明度可作为冗余编码，但图例必须写清，不能让同一种颜色同时含糊地表示强度和更新量。
   - “当前正在发生显著学习”的连接由真实瞬时更新客观确定，而不是为叙事手工挑选：

     \[
     \Delta J_{ij}
     =\Delta t\,\eta_Jr_i
       \left(R_j-\alpha_JJ_{ij}^{\beta}\right).
     \]

   - 建议在基础 `J` 连线上叠加独立 glow/outline：potentiation 与 depression 使用有符号 diverging colors，并在角落显示 Hebbian 项、decay 项及净更新的小型条形图或数值。
   - 高亮规则必须写入 animation config，例如显示 `abs(delta_J)` 的 top-k、超过全矩阵固定分位数阈值的连接，或与当前活跃 pre/post neurons 相连的 top-k；同一镜头内规则保持固定。
   - clipping 前后的更新应可区分。若某连接因 nonnegative constraint 被截断，动画标记为 clipped，不把零变化误读为没有 depression。

4. **Competitive-neuron state bands**
   - 第三层由上下对齐的两个主要节点行组成，同一列对应同一个 competitive neuron：
     - `U band`：膜电位 \(U_i(t)\)；
     - `V band`：适应变量 \(V_i(t)\)。
   - learning rule 的 postsynaptic factor 实际是 \(r_i(t)\)，因此还必须显示 firing rate。优先在 `U` 节点外增加由 \(r_i\) 控制的 halo/outer ring；若可读性不足，则增加一条较窄的 `r band`。不能让观众误以为权重直接由 \(U_i\) 更新。
   - 使用共享 preferred-position 横坐标，使 \(V\) profile 相对 \(U\) 的空间 lag 可直接观察；额外用细线标记 decoded centers \(\hat z_U,\hat z_V,\hat z_r\) 及 signed offsets。
   - `U`、`V`、`r` 是不同物理变量，分别使用带标签的 colorbar。颜色范围在一个 clip 内固定，不按每帧自动缩放，以免把幅值不变误画成增强或衰减。
   - 除节点颜色外，在 tutor curve 下方分别绘制连续的 `U profile` 和 `V profile`。competitive neurons 按最终 learned preferred position 排列；多个神经元落在同一位置时只在 profile 中取均值，离散节点仍全部保留，且不得为获得平滑外观进行空间滤波。
   - profile 同步显示 baseline-subtracted shape center，用于观察 `V` 相对 `U` 的 lag；原始 profile 本身不做基线扣除。

#### B. 时间尺度必须分镜表达

由于 \(\tau\ll\tau_v\ll1/\eta_J\)，单一实时速度无法同时清楚显示快速神经响应和缓慢权重形成。动画至少输出三个互补 clip：

1. `neural_dynamics.mp4`
   - 默认展示训练末尾约 20 s、包含多个 tutor passes 的连续窗口；时间接近神经动力学尺度。
   - 重点表现 \(R\to U\to r\)、\(V\) 的滞后以及一次局部更新的组成。
   - `J` 在该短窗口内可能几乎不变，这是正确的物理现象。可以用独立的 \(\Delta J\) overlay 显示微小更新，但不得人为放大后伪装成真实权重幅值。

2. `learning_evolution.mp4`
   - 将多个完整 tutor cycles 压缩成 learning-time montage，在固定 tutor phase 或每圈结束时抽取帧。
   - 重点展示随机 `J` 如何逐渐形成 diagonal/local Gaussian structure，以及 Gaussian correlation、width、update norm 如何同步变化。
   - 画面必须显示当前 cycle、累计物理时间和 playback acceleration，避免把压缩后的学习速度误认为神经时间尺度。

3. `global_training_dynamics.mp4`
   - 从保存的真实 `t=0` 初始状态均匀采样到训练末期，同时显示 `R/U/V/r`、top-k `delta_J` 连接和完整 `J(t)` 热图。
   - 使用独立的可调帧率配置；720 帧、30 FPS 可作为较快的建议基准，慢速教学版可进一步降低 FPS 或增加帧数。帧数决定时间抽样密度，FPS 决定播放时长，两者必须分别记录，不能把“只改 FPS”误写成增加了科学采样率。
   - 热图可以按最终 learned preferred position 重排 postsynaptic rows 以提高可读性，但 manifest 必须记录 permutation，且训练数组、连接索引与定量分析保持原始矩阵顺序。
   - 标注全训练进度、真实物理时间和平均 playback acceleration；科学变量不做帧间插值。

可增加一个短的 `single_synapse_explainer` 章节：暂停群体画面，选择由既定 top-k 规则确定的一个代表突触，用三项并列显示

\[
r_iR_j,
\qquad
\alpha_Jr_iJ_{ij}^{\beta},
\qquad
\Delta J_{ij},
\]

然后返回完整网络。这一镜头解释局部物理含义，但不得把单个突触的行为外推成整体收敛证据。

#### C. 可读性与规模控制

完整绘制 \(N_cN_{\mathrm{in}}\) 条连接在 \(N=128\) 时既不可读也开销过大，因此采用以下受控策略：

- 教学动画允许使用独立的较小 `animation_demo` 配置，例如 16--32 个 input/competitive neurons；必须明确标为 visualization-scale run。
- 对正式规模 run，只绘制按固定规则筛选的连接，并在独立、无重叠的 matrix panel 中显示完整 `J`，使未画出的连接不会被误认为不存在。
- 连接筛选只影响渲染，不影响 simulation、plasticity 或保存的权重矩阵。
- 禁止每帧重新选择色标范围。top-k 若导致连接频繁闪烁，可使用固定阈值加短时 visual persistence，但 persistence 只作用于 glow，不改变数据。
- 只对相机、标签和几何过渡做视觉插值；科学变量取自实际采样帧。若对变量做插值，必须在 manifest 中注明方法，且不得跨 tutor reset 插值。

#### D. 渲染架构与输出

- 动画必须离线读取 `training_history.npz`、`weight_history.npz` 和 resolved config，使同一个 run 可以重复渲染而无需重新训练。
- 为准确解释局部更新，history 至少保存动画采样点的 `time, tutor_position, R, U, V, r` 和相应 `J` snapshot；瞬时 `delta_J` 可由同一状态和配置确定性重算，或稀疏保存 top-k update events。
- `storyboard.py` 只定义章节、字幕、相机状态和数据帧映射；`feedforward_scene.py` 负责科学对象；`render.py` 负责 MP4/GIF/HTML 等后端，不把方程写进渲染循环。
- 优先输出 MP4；若本地缺少视频编码器，可输出 GIF 或 HTML 作为明确标注的 fallback，不应导致 simulation 失败。
- 每个动画伴随 `manifest.json`，记录 source run、时间窗口、采样/加速倍率、连接筛选规则、所有 color limits、插值方法、渲染版本和输出帧率。
- 字幕至少标出当前阶段是 `neural dynamics` 还是 `learning-time montage`，并在 feedforward-only 阶段注明“externally driven representation; not yet an autonomous attractor”。

#### E. 后续扩展约定

- 增加多层 feedforward hierarchy 时，沿相同 preferred-position 坐标增加 competitive bands；默认只展开当前相邻两层的显著连接，其余用 compact matrix inset 表示。
- 增加 recurrent `W` 后，不同时绘制所有 recurrent arcs。主视图使用动态 `W` matrix、aligned relative-displacement profile 和少量被选中的局部连接，避免形成不可解释的毛线团。
- recurrent 动画必须把 `J @ R` 与 `W @ r` 两类输入使用不同图例，并允许分别开关，以解释 tutor drive 和 recurrence 的贡献。
- recurrent 的全训练期 clip 同屏显示 `J(t)` 与 `W(t)` 两个固定色标热图，以及 `||delta_J||`、`||delta_W||` 轨迹；矩阵渲染可降低空间分辨率但不得降低实际训练矩阵分辨率。`joint_from_random` 与 `ff_pretrain_then_joint` 必须输出不同文件名和 manifest protocol 字段。
- M5 撤去 tutor 后的 autonomous test 使用冻结权重，不再显示 plasticity glow；应改为突出 `W @ r` 和 bump survival/motion，避免让测试阶段看起来仍在训练。
- path-integration 阶段增加 uniform `I_speed(t)`、bump width 与 decoded speed 三条同步轨迹，明确展示：

  ```text
  speed current -> baseline shift -> bump width -> bump velocity
  ```

#### F. 动画验收标准

- [ ] 任一显示的节点、连线和数值均可追溯到 source run 的数组及 index。
- [ ] `J` 强度、potentiation/depression 和 `r` 不共享含糊的颜色语义。
- [ ] `V` 相对 `U/r` 的 lag 可在代表性参数下被观察并由静态指标验证。
- [ ] 高亮连接与数值计算得到的 `delta_J` top-k/threshold 完全一致。
- [ ] 切换 `learning_enabled=false` 后不显示虚假的 update glow。
- [ ] `m=0` control 中 `V` 的行为与方程一致，动画不保留旧状态造成的假 lag。
- [ ] 降低渲染帧率或改变 display top-k 不改变 simulation metrics。
- [ ] 动画中的 Gaussian、节点颜色和动态 matrix 在抽查帧上与保存数组一致。
- [ ] 三个时间尺度 clip 均标注真实时间与播放倍率。
- [ ] README 说明如何从已有 run 重渲染动画及其教学用途边界。

绘图约定：

- 指标图必须同时显示 bump-validity mask，避免对 collapse 状态解码。
- 理论预测、自由拟合和 simulation measurement 使用不同线型并在图例中写清。
- periodic 与 paper-reset 结果不得混在同一条无标注曲线上。
- 默认保存 PNG；关键汇总图可同时保存 PDF/SVG，但不以矢量格式替代原始数值。

## 9. 测试与数值健康检查

### 9.1 单元测试

- normalized Gaussian 的离散积分接近 1。
- `J @ R`、`W @ r` 的 shape 和 pre/post 方向正确。
- divisive inhibition 分母为正，rate 非负且有限。
- 单步 Euler 更新与手算小例一致。
- 单步 plasticity：共激活增强、衰减符号正确、元素幂不是矩阵幂。
- nonnegative clipping 只在配置启用时执行。
- `eta=0` 时权重不变；`m=0` 时 adaptation 衰减至零。
- circular decoder 跨边界连续；linear decoder 不被用于 periodic 数据。
- equilibrium width 在 \(\beta=0.5\) 给出 \(\sigma_J=\sigma_R\) 和 \(\sigma_u/\sqrt2=\sigma_R\)。

### 9.2 集成与 smoke tests

- 极小网络短跑可以完成、保存并重新加载。
- 相同 seed 结果一致，不同 seed 初始权重不同。
- `dt` 减半后固定物理时长的轨迹与关键指标接近。
- resume 若实现，必须从完整 checkpoint 继续且不重复时间点。
- divergence、NaN、无效 bump 会使 run 标记失败或指标 masked，而不是继续生成误导图。

### 9.3 科学验收而非“看起来像”

不能仅凭权重热图出现斜带就宣布复现成功。M2 至少同时满足：

- 多 seed 下对齐 profile 接近 Gaussian；
- width 对 \(\beta\) 的依赖方向和理论一致；
- translation-invariance error 随学习下降；
- weight update norm 进入稳定区；
- 活动保持局域且可解码；
- 禁用学习或破坏均匀经验后，上述结构显著改变。

M3 至少同时满足：

- shift 对运动方向有正确符号；
- `m=0` control 显著减弱 shift；
- 排除边界 reset window 和启动 transient；
- `dt` convergence 与多个 seeds 支持结论。

## 10. README 最低内容

实施时 `prospective/README.md` 至少包含：

- 研究问题和“一句话机制链”。
- 对应论文与本地 PDF/notebook 路径。
- paper-faithful 部分、toy 简化和当前未实现部分。
- 环境安装、可复制的训练/分析/测试命令。
- 方程到代码模块的映射表。
- 配置字段语义，特别是 Gaussian amplitude、density 和矩阵方向。
- run 输出结构和每个核心指标的解释。
- 已知数值风险、失败模式和当前复现状态。
- 不将 feedforward prospective coding 误称为 autonomous continuous attractor 的声明。

## 11. 建议实施顺序

- [ ] 人工确认本 TODO 的科学范围、默认边界条件和首轮参数策略。
- [ ] 实施 M0，并仅运行测试。
- [ ] 实施 M1，先验证固定理论 `J` 下的动力学。
- [ ] 实施 M2，观察局部 learning rule 的核心效果。
- [ ] 基于已保存的 M1/M2 run 实施 `feedforward_mechanism` 动画，并逐帧抽查变量与更新量。
- [ ] 实施 M3，验证 prospective coding 与关键 controls。
- [ ] 汇总 M0--M3 的结果与偏差，再决定是否批准 M4--M6。
- [ ] M4--M6 完成后，才讨论与 Clark/Vafidis 的系统比较。

## 12. 审阅时需要确认的决策

在开始写代码前，请重点确认：

1. 首轮是否同意只实施 M0--M3，把 recurrent attractor 和 PI 延后？
2. 主实验是否坚持论文的 `paper_reset` 线段，把 periodic ring 只作为控制？
3. 参数策略是否采用“论文代表参数 + 明确命名的较小 toy 参数”双配置，而不是只保留一套？
4. 是否接受多 seed 和 `dt` convergence 为必需验收，即使它们增加运行时间？
5. M2 的成功标准是否以 profile、width law、translation invariance 和 learning convergence 的联合证据为准？

确认这些决策后，再依据本 TODO 创建代码；实施过程中若必须偏离论文方程，应先更新本文件并说明原因，而不是在代码中静默加入修正。
