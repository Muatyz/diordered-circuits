# Ságodi 2024 对 `learning` 代码库的设计指导

## 结论

当前代码库不需要改变 Vafidis predictive local plasticity 的学习规则；需要改变的是训练后分析的核心抽象与验收标准。第一目标应从“得到严格的 continuous fixed-point ring”调整为“得到可验证的 approximate continuous attractor”：一个 autonomous、闭合、吸引的 slow invariant ring，其法向恢复远快于切向漂移，并能在明确的行为时间窗内满足可测的记忆误差界。

## 对当前实现的审计

### 可以保留并扩展

- frozen testing 与 training 分离符合论文分析前提。
- `bump_attractor_trajectory_history.npz` 的多初始角 darkness 轨迹可作为 slow-set identification 的原始数据来源。
- `analysis/phase_flow.py` 已有周期 scalar flow、stable/saddle roots 与 basin boundaries，和论文 S7.6.3 最接近。
- trajectory-based normal/tangential timescale assay 是有价值的非线性补充。
- 多 seed、neuron count、noise ensemble 的实验框架可直接承载新的 slow-manifold 指标。

### 必须收紧的解释

1. `hd_tuning_history["v_hd_distal"]` 是 visual-teacher steady response curve，不自动等于 autonomous invariant manifold。
2. `weight_eigenvalues.npz` 不是动力学 Jacobian spectrum，不能证明 normal hyperbolicity。
3. 当前 manifold distance 只在 `v_hd_distal` 中计算；Ságodi 的几何与 Jacobian 条件要求完整 frozen Markov state。
4. PCA 只能展示 full-state 分析结果，不能用前三主成分中的视觉距离替代 full-state distance。
5. gain≈1、低 diffusion、低 bump drift 与 ring-like weight 都是性能或结构证据，但单独都不是 approximate continuous attractor 的充分条件。

## 建议的分析数据模型

每个 frozen autonomous anchor 至少保存：

```text
theta_decoded
state_vector
state_flow
tangent_vector
tangent_speed_rad_s
normal_flow_norm
jacobian_eigenvalues
slow_eigenvector_tangent_alignment
normal_spectral_margin
nearest_manifold_residual
```

每个网络汇总保存：

```text
ring_closure_error
angular_coverage_fraction
eta_theta_rad_s
normal_spectral_margin_min
spectral_gap_min
state_perturbation_recovery_fraction
stable_fixed_point_count
saddle_fixed_point_count
basin_entropy
max_basin_width_rad
finite_horizon_bound_pass_fraction
```

## 模块边界

- model step 继续负责可训练模型，但新增纯 frozen autonomous wrapper，避免分析代码伪造模型动力学。
- slow-manifold identification、Jacobian、phase flow 各自成为独立 analysis 模块。
- experiment 入口只编排和保存；plotting 只读取 `.npz/.json`。
- configuration 将 state perturbation 与 weight perturbation 分成两个命名清晰的组。

## 推荐实施顺序

### P0: 建立可信动力学接口

定义 canonical frozen state/map，测试它与现有 zero-input、visual-off、training-off step 完全一致。先解决 `r_hr` 一步 lag 是否属于状态的问题，再计算任何 Jacobian。

### P1: 识别 autonomous slow ring

从长 darkness trajectories 的慢速尾段建立 full-state periodic curve；输出 closure、coverage、invariance residual，并保留现有 teacher-manifold assay 作为独立对照。

### P2: normal hyperbolicity

沿 ring 使用 finite differences 计算 full flow Jacobian。禁止使用 autograd；这不影响项目的“无 backprop”约束。报告一个 tangent slow mode、其 tangent alignment、第二大实部以及全环最差 spectral margin。

### P3: finite/asymptotic memory

从 zero-input phase flow 得到 `eta_theta`，验证 circular error 的 finite-time bound；从 stable/saddle roots 得到 basin fractions、entropy 和 asymptotic worst-case error。

### P4: 鲁棒性与 scaling

将现有 noise/neuron-count grid 的纵轴扩展为 `eta`、spectral margin、basin entropy 与 bound pass rate。另建 D-type frozen-weight perturbation sweep，不能用 training visual noise 代替。

### P5: 展示

在 full-state 结论成立后画 PCA 前三维轨迹、slow ring、perturbation return 与固定点；图中同时注明 PCA explained variance，所有定量距离仍在 full state 计算。

## 对 TODO 未完成项的直接回答

- “计算 Jacobian 及 eigenvalue spectrum”应升级为 frozen full-dynamics Jacobian along autonomous manifold，而不是单个 state 或 weight spectrum。
- “PCA 前三主成分 perturbation analysis”依赖 P1/P2，不应先做。
- “neuron count 性能比较”应比较 slow-manifold 指标；有限大小导致离散 stable/saddle points 本身不是失败。

## 不采用的内容

论文使用全局 MSE、Adam 和 backprop 训练通用 RNN。该部分只作为“哪些 dynamical solutions 会出现”的证据，不移植到本项目 learning rule；本项目继续只允许局部变量驱动的 Vafidis-style plasticity。

## 2026-07-27 implementation and first audit

已实现：

- canonical state `[r_hd_to_hr_lp, r_hr, i_hd_distal, v_hd_distal]`；
- exact frozen map、Euler-equivalent flow 与 analytic Jacobian；
- 逐轨迹 `10^-3 max ||f||` slow-point capture；
- decoded-angle coverage gate 与 periodic cubic spline；
- coordinate-correct `theta_dot`、flow reversal、basins/entropy；
- leading Jacobian modes、normal margin、spectral gap 与 tangent alignment；
- focused saved-run CLI 和 accepted/rejected 两种诊断图。

首个正式 run `20260727-144656_vafidis_toy_42` 的 1024 个 slow candidates 只覆盖
21/180 angular bins（11.7%），形成 21 个 disconnected low-speed angle clusters。
因此 50% coverage gate 拒绝了 ring spline，并禁止输出该假想 spline 的 roots 和
Jacobian。结果与“当前网络更像离散/pinned attractors”一致，但这些 clusters 仍只是
slow-state evidence；确认 stable/saddle fixed points 需要下一步在各 cluster 之间增加
中速过渡采样或直接做 local flow-reversal probes。
