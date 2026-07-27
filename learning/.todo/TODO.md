1. 项目目标: 验证 Vafidis-style predictive local plasticity 是否能够学习 Head direction bump maintenance 和 (angular) velocity-driven path integration

2. 变量命名: 严格对应 notebook 中的物理量, 确保代码的可读性

3. data stream: 每个 timestep 需要按照固定顺序更新. 

    真实 $\theta, v$; visual/velocity input; HR dynamics; HD distal dynamics; proximal voltage; firing rate; learning error; PSP traces; weight update

4. 训练和测试分离: training phase 有 visual teacher 并 update weights; testing phase 冻结权重并且移除 visual teacher, 只保留短暂 cue 或直接完全 darkness

5. 数值积分方式: 先使用 Euler method 作为 baseline 实现. ODE 共用微分时间 dt, 不混用离散更新和连续时间公式

6. 诊断图与指标: 输出并保存 weight matrix, HD activity heatmap, decoded heading v.s. true heading, PI error, velocity gain curve, bump maintenance trace

7. 成功标准: 检查 $W_{HD\to HD}$ 是否形成 local symmetry, $W_{HR\to HD}$ 是否形成左右相反的不对称偏移结构, $\hat{\theta}$ 是否能在 darkness 下近似积分 $v(t)$

8. 该项目是验证 local learning rule 的, 因此严禁引入 back propagation, PyTorch autograd, global loss optimization, RNN trainer, supervised regression 等方法. 学习规则仅使用 Vafidis 论文中提到的局部变量方法. 

## 2026-07-02 current diagnostic status

- Added PVA and peak/plateau decode traces. `theta_hd_decoded` remains PVA; `theta_hd_decoded_peak` is the highest-peak diagnostic after collapsing paired HD cells.
- Activity plots now use `[-pi, pi]` axes and label both decode methods.
- The previous `~6 deg` bump-maintenance offset was caused mainly by a broad, saturated flat-topped bump, not sustained zero-velocity drift.
- Default visual width now follows the release-code `sigma = 0.15` mapping, `kappa = 11.11111111111111`.
- Verification run: `runs/vafidis_toy/codex_kappa11_peak_decode`.
- Remaining issue: the narrower visual teacher fixes bump offset and tuning slices, but darkness velocity is under-gained by about `1 rad/s` at the 500 deg/s test, so the next target is HR-to-HD protocol/calibration.

## 2026-07-02 follow-up after plotting audit

- `darkness_hd_activity_heatmap.png` is now explicitly labeled as visual-off but velocity-input-on PI testing. It is not the zero-input drift test.
- Zero-input drift should be read from bump maintenance: current fitted drift is `3.84e-6 rad/s` (`2.20e-4 deg/s`).
- `darkness_pi_error.png` now uses radian units and a fixed `[-pi, pi]` axis.
- Added `training_heading_short_window.png`, a two-panel short-window comparison of network decode and true heading.
- `gain_velocities` now samples 11 points from `-500` to `500 deg/s`.
- New run: `runs/vafidis_toy/codex_gain11_semantic_figures`, with `velocity_gain = 0.9721`.
- Remaining unsolved issue: high-speed PI at 500 deg/s is still under-gained by about `1.00 rad/s`; fix should follow paper-compatible gain adaptation / longer training / smaller `dt`, not post-hoc supervised calibration.

## 2026-07-02 follow-up after spectrum / annotation update

- Activity slice and short heading-window figures now label the exact source time interval from the original heatmap/history.
- `velocity_gain_curve.png` annotates fitted PVA and peak gain values directly on the figure.
- Added combined `training_weight_matrices_side_by_side.png` for `W_HD->HD` and `W_HR->HD`.
- Added eigenvalue diagnostics: `training_weight_eigen_spectrum.png`, `weight_eigenvalues.npz`, and `weight_spectrum_diagnostics.json`.
- Current `W_HD->HD` shows approximate nonconstant-mode double degeneracy (`86.2%` of sorted-real adjacent pairs within a 2% normalized gap), so the recurrent operator is more ring-like than the remaining activity flat tops suggest.
- Current interpretation of large darkness PI error: not faithful evidence that the original Vafidis local rule is intrinsically flawed. It is an incomplete toy/protocol reproduction, because the paper reports near gain-1 darkness PI over `|v| < 500 deg/s`; our latest run still has a `-1.0038 rad/s` high-speed bias at `500 deg/s`.
- Remaining flat-top tuning slices are real sigmoid/recurrent saturation in the reduced toy, not just the old paired-angle plotting bug. Future fixes should target reduced-dynamics parameters or Vafidis-style gain/protocol adaptation.

## 2026-07-03 V-D-V PI protocol and peak-readout diagnostic update

- PI tests now use an explicit `pi_cue_duration`, separate from the short bump
  maintenance `cue_duration`.  The default timing is `4 s` visual, `6 s` dark,
  and `2 s` visual re-cue, matching the 20:30:10 proportions of the released
  Figure 2A / Appendix 1 example.
- Testing remains frozen-weight: visual segments provide the paper's teacher
  input, but `training=False` throughout bump, constant-velocity PI, OU PI, and
  velocity-gain probes.
- `training_heading_short_window.png` now overlays true heading, PVA decode,
  and peak decode in one panel, with circular decode error in the second panel.
- Constant-velocity and OU path-integration heading plots use `pi rad` units on
  the y-axis, so high-speed tests no longer appear as thousands of degrees.
- Added `*_saturated_hd_bins` metrics after paired-HD angular collapse.  These
  quantify the flat-top failure mode that makes peak decode unreliable when
  several adjacent angular bins sit near the sigmoid ceiling.

## 2026-07-03 activation / peak-sharpness audit

- Directly changing the sigmoid gain/bias can reduce saturated plateaus, but
  the short retraining probes showed a clear tradeoff: lower gain or higher
  bias damaged bump maintenance and/or velocity gain before producing a robust
  single-bin peak.
- Narrowing the visual teacher also makes the peak more unique, but this
  departs from the released-code `sigma = 0.15` mapping and strongly degrades
  PI gain in the toy.
- The current safe code change is therefore diagnostic/readout scoped: peak
  decode now groups bins within the same near-saturated peak top using a 0.5%
  tolerance, and `*_near_peak_hd_bins` is saved beside `*_saturated_hd_bins`.
- A genuine single-peak tuning curve likely requires retuning the toy's voltage
  and learning-rate scale, not adding nonlocal winner-take-all, HR mirroring,
  or post-hoc supervised calibration.

## 2026-07-06 bump-maintenance peak-decode audit

- The latest `attractor_noise_comparison_baseline` run's peak decode error is a
  one-bin offset (`-12 deg`), not a continuous zero-velocity drift.  At visual
  cue offset the peak is still at `0 deg`; during darkness release the activity
  relaxes into a neighboring flat-topped attractor basin centered near
  `-12 deg`.
- The final collapsed HD rates have near-peak bins at `-24, -12, 0 deg`, with
  the highest bin at `-12 deg`.  Therefore a peak decoder cannot honestly return
  `0 deg` without using the cue/target as post-hoc information.
- Activity heatmaps now collapse paired odd/even HD partners before plotting on
  the theta axis.  This removes the previous misleading 60-row display where 60
  HD cells were shown as 60 independent angular positions even though the model
  has 30 paired angular bins.
- Bump metrics now include release-shift diagnostics, so future runs can
  distinguish cue-release basin jumps from late-time fitted drift velocity.

## 2026-07-06 default-training overrun audit

- Replaying saved `weight_history.npz` snapshots showed that the default seed's
  final `4080 s` total training state was worse than the `4000 s` snapshot:
  darkness RMS PI error was `1.426 rad` at `4080 s` but `0.077 rad` at
  `4000 s`; bump final drift was `0.216 rad` at `4080 s` but `0.00031 rad` at
  `4000 s`.
- The previous `80 s` recurrent-only warmup was removed because it is not part
  of the release-code training protocol.  The default config now uses a single
  `4000 s` OU training phase with visual teacher present and all plastic
  weights updated from `t=0`.  The high-velocity HD-to-HD learning gate was
  removed for the same reason.
- Verification with the default seed after this protocol change:
  `velocity_gain = 0.9955`, `darkness_rms_pi_error = 0.0772 rad`,
  `darkness_final_abs_pi_error = 0.1136 rad`,
  `bump_final_abs_drift = 0.00031 rad`, and peak bump drift `0`.

# 7.3 meeting notes

目前的 peak decode 是识别最高点作为 heading direction, 目前误差已经降至最低以便后续代码工作. 如无必要不要再修改! 如果后续在实验的时候发现 peak decode 的误差甚至已经高于 PVA decode 则立即停止代码文件的编写

## 1

- [x] visual input 的形式
    - [x] 更换为 + noise 形式(优先完成)

      还能够学到 quasi-continuous attractor 结构吗? 如果可以, 探索不同 noise 参数对于 attractor structure 的影响 (比如绘制 noise 参数为横轴, 纵轴是一些对网络性能评估的参数)
    - [x] *tuning curve sampled by Gaussian generative process(作为候选, 不一定一定要完成)


- [x] visualize: 模仿 Clark 的在 target manifold 附近设置微扰初态, 通过前三维 PCA 进行可视化, 同时追踪微扰态到流形最近距离随时间的变化
    - [x] 首先是在 visual teacher 下的观察;
    - [x] 完成训练之后, 在 darkness 下的某时间 trajectory;

    注意:  target manifold 原始定义为连续流形, 由于计算机精度因此需要处理为大离散点集, 在计算态到流形的最近距离需要考虑到这一点; 若有不太清楚的地方可以参考 /reproduction 子项目中的相关代码
- [x] 延长 darkness 后 re-visual input 的时间, 观察 error 是否能重新跌落回到 0
- [x] 尝试减少 neurons 数量(比如更接近真实果蝇的 16), 对比不同神经元数量对网络性能参数的影响, 从而评估子项目采用 learning rule 的鲁棒性. (绘制图像进行对比)

实现入口: `python -m learning.experiments.run_attractor_robustness --config configs/experiments/vafidis_toy.yaml`
## 2

- [x] 绘制权重矩阵跟随 training time 的变化, 观察什么时间点权重矩阵实际上已经收敛, 避免过度训练浪费时间
- [x] 重新梳理 figures 文件夹的结构, 避免所有图全部混在同一个文件夹
- [x] 尝试解决有关 diffusion approximation of diffusion phenomena 的问题, 并且解出具体的 diffusion coefficient
- [x] gain 曲线需要绘制 in darkness 下的情况, 参考原文献中 figure 的 setup.

实现: 默认 `visual.noise_std = 0.0`, `model.n_theta = model.n_hr = 60`; `weight_history.npz` 保存 sparse weight snapshots; figures 按 `activity/`, `heading/`, `weights/`, `gain/`, `diagnostics/` 分组; diffusion coefficient 保存到 `test_metrics.json`; noise / neuron-count 比较接口见 `configs/analysis/noise_comparison.yaml` 和 `configs/analysis/neuron_count_comparison.yaml`.

## 3

- [x] 更高时间分辨率的 weight matrix development 绘制. 目前的采样时间间隔太大, 以至于无法更细致地观察 weight matrix 的收敛过程. 可以将其添加至 config 文件中从而进行细调;
- [x] 延长 noise × neuron-count meshgrid 的 noise std list。此前 n=32 出现了随 sigma 增大、网络性能反而增强的现象，因此先用宽范围 std 轴定位该增强何时衰减或反转（噪声相对 visual 幅值不可能无限增益），再固定 n=32、在候选 critical std 附近使用更密的 std list，观察临界行为如何出现，并为后续理论分析保留 noise-to-signal ratio 与相对 sigma=0 的配对性能变化；
- [x] 重新实现 bump drift 的计算, 看一下如何实现 diffusion coefficient 的计算. 可以对训练完成的 weight matrix 进行多次(>100? 这个数字可以先通过单个网络来确定) darkness 下的 bump 位置 trajectory 采样跟踪, 然后进行系综平均. 也可采用 Vafidis 原论文中的实现方式. 在实现前需要先对实现方法进行说明

实现: weight snapshot 间隔由 config 调节，并包含精确的 `t=0` 与终点；宽 meshgrid 仅作为单 seed 探索。由于 n=16/24/32 在 `sigma=0.02/0.05/0.30` 呈现非单调成功/失败切换，新增 `noise_by_neuron_count_low_n_replication.yaml`：预先固定四个 sigma、三个 neuron count 和十个配对 seed，并将 noise 限制在训练阶段，统一使用无噪声 visual test 与 darkness test，避免训练效应和测试噪声混杂。只有效应跨 seed 复现后，才运行 `noise_by_neuron_count_n32_critical_scan.yaml` 对 `0.02–0.05` 与 `0.25–0.35` 两个局部窗口细扫。每个 neuron count 均输出原始 seed trajectory、相对 sigma=0 的 paired delta、median 与 95% bootstrap CI。bump diffusion 的 trial 数、时长和 test noise 均由 config 调节。按照 Vafidis Eq. 21 使用 `D = Var(Delta theta) / T`，同时单独保存 `mean(Delta theta) / T` 作为 systematic drift，避免把确定性 side bias 计入 diffusion。


## 4

- [ ] 重新考虑 std 和 visual cue amplitude 的 scaling effect
- [x] 考虑不同的视觉 cue profile, 以及考虑引入 heterogeneous visual cue generator 的可能性

  实现: `configs/experiments/vafidis_mammalian_heterogeneous.yaml` 当前使用 240 个 HD/HR 神经元；visual teacher 采用 `/reproduction` 中 Clark wrapped-Gaussian process + normalized softplus 的复现参数，每条静态异质调谐曲线按 circular COM 对齐到对应 HD preference。默认保留逐神经元 unit-angular-mean normalization，再让所有 generated curves 乘同一个 shared amplitude；完整 teacher realization 随训练权重保存，便于复现和诊断。
  诊断更新: sampled teacher profile 已标记 preferred orientation；恒速和 OU activity heatmap 均标记 dark phase；heterogeneous gain scan 加密为 41 点并连接原始响应，低于 `R²=0.95` 时不再绘制误导性的线性拟合。测试指标新增线性拟合 `R²`/RMSE、连续可积分速度工作区间，以及 cue/dark population slice 的局部峰数量。现有 run 表明，单细胞 teacher 通常为 1–3 峰，而 activity slice 的更多峰来自固定 heading 下跨异质细胞的 population cross-section；`500 deg/s` 锁定与 OU 可积分则对应非线性的有限速度工作区间，不能仅用全局 gain 判定。因此恒速 PI 展示速度改为近单位增益区间内的 `75 deg/s`，而 41 点 gain scan 仍保留 ±500°/s 作为失效边界压力测试。

  幅值与展示修正：sampled heterogeneous visual input figure 默认抽取 16 个神经元并固定为 4×4。最终撤回 `global_peak` 和默认 `per_neuron_peak`，回到 Clark-faithful `unit_angular_mean` generated curves，并统一乘 shared amplitude。shared amplitude 取 `4 * exp(-kappa) * I0(kappa) = 0.484419041134548`，严格匹配原 Vafidis peak-normalized cue 的环上平均 excitation；因此保留 peak-height heterogeneity、不依赖 `N`/seed 极值，也避免把原 amplitude=4 直接乘 unit-mean curves 导致过强输入。

# 7.20 meeting notes

## 1

- [x] 绘制单神经元的 tuning curve
- [x] 在计算 path integration error 的时候, 考虑对多条 trajectory 进行 ensemble average, 以观察是否存在平均的 drift velocity(去除 peak decode 而只使用 PVA/COM decode, 因为 peak decode 误差在 heterogeneous 中太大了)
- [x] 核实 across neuron 的 activity profile 是否已经进行了 sort by PVA/COM, 以排除多峰的出现是不是由画图错误引起的
- [x] 同样地, 核实 weight matrix 是否已经进行了 sort, 从而确认 sparse weight matrix 是画图错误还是建模本身特性
- [x] 增加 zero-velocity 的 n 值以增强平均效果

实现：新增 frozen-weight heading sweep，直接保存并绘制单细胞 firing-rate tuning curve，使用各细胞 tuning curve 的 circular COM 作为经验 preferred direction。标准 activity heatmap/slices 按该经验 COM 排序；权重主图对 HD target/source 显式应用同一 permutation，HR source 保持 L/R wing 分块并在 wing 内排序，同时额外输出 raw-index 矩阵。矩阵排序前后零元素数量严格不变，并新增 near-zero fraction 指标，因此 sparse structure 可与绘图顺序区分。新增 24 条独立 OU PI trajectory 的 PVA-only error ensemble，输出 mean/SEM 和平均 drift velocity；zero-velocity diffusion ensemble 移除 peak decode，并将 heterogeneous preset 的 trials 从 120 增至 300。

Weight matrix 坐标修正：COM 仅作为 target/source permutation 的排序键；最终矩阵、并排矩阵及训练 snapshot 均以排序后的 neuron ID 为横纵轴，每个神经元保持相同像素宽度，不再把非均匀经验 COM 强行映射到 `[-pi, pi]`。实际 COM 数值仍保存在 `empirical_com_sort_order.npz` 中供复核。

绘图修正：经验 COM 并非均匀角网格，不能用 `imshow` 将排序行号等距映射到 `[-pi, pi]`。activity/visual-current heatmap 现按实际 COM 中心绘制；activity heatmap 的 PVA/peak overlay 也在同一经验坐标内重算。该修正只改变诊断图，不改变网络状态、学习权重或保存的标准模型坐标解码历史。

快速审计（现有 n=240 权重，降低角度/settling 分辨率，仅用于实现核查）：经验 COM permutation 移动了约 92.9% 的 HD 单元，说明旧 activity 图并未按学习后的 COM 排序；排序不会改变任何矩阵元素。以各矩阵最大绝对权重的 1% 为 near-zero 阈值时，HD→HD 与 HR→HD 比例分别约为 2.39% 与 0.57%，因此旧图的“稀疏感”不是由大量严格零连接造成。正式解释应以新配置的完整 120-angle tuning sweep 和 raw/sorted 对照图为准。

## 2

- [ ] 引入 Clark 文章中的 attractor analysis method. 部分具体的实现方式可参考 /reproduction 相关代码
  - [x] 绘制根据幅值归一化后的 COM aligned tuning curve 以及 std curve, 使用粗黑实线表示平均值
  - [ ] 计算该建模体系下的 Jacobian matrix 及其 eigenvalue spectrum
  - [ ] 计算 PCA 前三主成分的可视化, 进行 perturbation analysis
  - [ ] 进行 neuron count 的性能比较

  实现：对 frozen-weight heading sweep 中全部 `N` 个 HD 神经元分别除以自身最大 firing rate（不使用全局 maximum），再按 circular COM 以整数 heading bin 对齐到 0（不插值、不抽样）。当前一个训练网络定义为一只模拟小鼠，因此图中 mean/std 是该模拟小鼠内部跨 `N` 个神经元的统计量，不是 Clark Figure 2E-F 中 31 只小鼠的跨小鼠平均。保存 peak-normalized aligned matrix、逐神经元 peak、COM/shift 及 within-mouse mean/std 到 `hd_tuning_com_aligned.npz`，同时保留 Clark 原文使用的 unit-mean 数值版本作为参考；`figures/activity/com_aligned_hd_tuning_population.png` 左图绘制全部 `N` 条 peak-normalized 曲线并以粗黑实线叠加 within-mouse mean，右图绘制 `ddof=0` 的 within-mouse neuron std curve。

  Heterogeneous setup 修正：旧 `0.35 s` fixed-heading sweep 未达到稳态，完整 120-angle 审计在延长至 `1.4 s` 后 firing-rate matrix 仍变化约 31.5%。heterogeneous preset 现采用最短 `1.4 s`、最长 `6.0 s`，并要求最近 `0.2 s` 的最大 HD-rate change 不超过 `0.002`；逐 heading 的实际 settling time、convergence flag 与残差均保存并单独画图，未收敛方向不得静默解释成 steady tuning。新增 visual-only 与 post-training steady tuning mean/std 对照，避免把 teacher/activation 与 recurrent learning 混为一谈。一个固定 `N`、独立 seed 的网络仍定义为一只模拟小鼠；新增 10-seed `heterogeneous_simulated_mouse_replication.yaml`，按 Clark 层级先算鼠内 mean/std，再画单鼠细线与跨鼠粗黑平均。同时 neuron-count setup 扩展为五个 paired seed，只作为 `N` scaling 诊断，尚不修改 Vafidis local rule。Heterogeneous teacher 样本图由 8 条扩展为 16 条，并使用 4×4 子图布局。

  Silent-neuron 容错与诊断：post-training tuning 中 angular mean 或 peak 不超过 `1e-12` 的 HD 细胞不再导致 COM-aligned figure generation 整体失败。它们以零曲线保留在 all-`N` peak-normalized mean/std 中，unit-mean 行标为 NaN，并显式保存 valid/silent mask、count、fraction。该容错只修复分析流程；silent fraction 仍代表 recurrent learning/dynamics 的失败模式。

  30-mouse hyper replication：新增 `heterogeneous_30_mouse_hyper.yaml`，固定每只模拟小鼠为 `N_HD=N_HR=120`，用 base seed 42 加 offsets `0..29` 独立训练 30 个网络。首轮 hyper-level 输出只保留逐神经元 peak-normalized、COM-aligned 的层级图：每只鼠先跨 120 个 HD 神经元计算 mean/std，以浅灰细线绘制，再对 30 只鼠等权平均并以粗黑线绘制；generic neuron-count metric 图与 unit-mean 版本暂时关闭，待检查数据处理效果后再扩展。归档保留每只鼠的 convergence fraction；首轮 audit 不静默排除未收敛小鼠，但解释结果时必须显式检查该 QC 字段。

  Clark Figure 4 A-C hyper diagnostic：A/B 在 `[-pi,pi]` 上分别绘制逐鼠 empirical tuning COM 黑点和保持逐鼠点数相同的 circular-uniform null。C 对小鼠 `1/10/20/30` 使用同一随机排列的嵌套 `N_sub=5/10/20/40/80/120` 子集，按 Eq. 4 计算未减均值的 `C(theta,theta')=<f_i(theta)f_i(theta')>_i`。计算前撤销 mean/std 图使用的逐曲线 COM alignment，避免人为消除或制造 translation symmetry；每格同时报告到 circulant projection 的相对 Frobenius error，并保存完整矩阵 NPZ 与误差 CSV。有限样本误差不要求逐列严格单调，重点比较随 `N_sub` 增长的总体趋势，并以 120 个有效 HD 神经元的全体矩阵作为终点。

  COM uniformity inference：A 图对每只模拟小鼠的有效 empirical COM 执行 one-sample Kuiper circular-uniformity test，并对 30 个 p-value 进行 Benjamini–Hochberg FDR correction（默认 `alpha=0.05`）。图标题报告 raw 与 BH-corrected rejection count；逐鼠 statistic、p、adjusted p/q 与 reject flag 保存为 `heterogeneous_clark_figure4_com_uniformity.csv`。未拒绝 null 不等价于证明 COM 严格均匀生成。

# 2026-07-27 Ságodi slow-manifold design guidance

- [x] 摄取 `Back to the Continuous Attractor`，建立 card/equation/figure/code/open-question 映射与 learning 设计说明。
- [ ] P0：定义 frozen autonomous minimal Markov state、pack/unpack、one-step map 与 flow；先审计当前 `r_hr` 一步 lag，再开始 Jacobian 实现。
- [ ] P1：从多初始角、零输入、无噪声、冻结权重的长 darkness trajectories 识别 full-state periodic slow manifold；保存 closure、coverage 与 invariance residual。现有 visual-teacher `v_hd_distal` curve 保留为 teacher-manifold 对照，不再当作 autonomous manifold。
- [ ] P2：沿 autonomous manifold 计算 finite-difference full-dynamics Jacobian。报告 slow eigenvalue、第二大实部、spectral gap、normal margin 与 slow eigenvector/tangent alignment；`weight_eigenvalues.npz` 不得替代该诊断。
- [ ] P3：扩展 zero-input phase flow，保存 `eta_theta=max|dtheta/dt|`，验证 finite-horizon circular-memory bound；加入 stable/saddle count、basin fraction、basin entropy、最大 basin 宽度和独立长轨迹核验。
- [ ] P4：把 S-type state perturbation 与 D-type frozen-weight perturbation 分成独立配置/结果；training visual noise 不视为 D-type robustness。
- [ ] P5：只有在 full-state slow manifold/Jacobian 完成后再做 PCA 3D perturbation 图；PCA 仅用于展示，定量距离仍在 full state 计算。
- [ ] 将 neuron-count/noise 比较的主要纵轴扩展为 `eta_theta`、全环最差 normal spectral margin、basin entropy 与 bound pass rate；有限网络出现离散 fixed points 不自动判失败。

详细设计：`learning/reports/notes/sagodi2024_learning_design_guidance.md`。文献实现契约：`references/sagodi2024_back_to_continuous_attractor/`。
