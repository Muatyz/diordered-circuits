# Figure Map: sagodi2024_back_to_continuous_attractor

## Figure 2: perturbed ring attractors

### Paper meaning

有限神经元数或小的参数扰动通常把理想固定点连续体变成含 stable/saddle pairs 的 invariant ring，而不是把 ring geometry 完全摧毁。

### Current code relation

- `learning/src/learning/analysis/phase_flow.py`
- `bump_attractor_trajectory_history.npz`
- `velocity_phase_flow_summary.npz`

### Reproduction criterion for this project

- zero-input scalar phase flow 覆盖完整环。
- roots 在周期边界正确处理，并尽量呈 stable/saddle 交替。
- 长轨迹的终点与独立 phase-flow root/basin 预测一致。

### Status

Implemented conditionally：已有 full-state slow-point capture、coverage gate、periodic spline、reversal roots 和 basin entropy。首个 `N=120` run 因低速点仅覆盖 11.7% 环而正确拒绝 spline；因此该 run 尚不能给出 ring-flow roots。

## Figure 4E-F: eigenspectrum along the slow manifold

### Paper meaning

沿流形各点的动力学 Jacobian 有一个 slow tangent mode，并与较快的 normal modes 分离。

### Current code relation

- `weight_eigenvalues.npz` 只包含 effective connectivity spectrum。
- `timescale_separation_history.npz` 提供 trajectory-based operational timescale ratio。

### Reproduction criterion for this project

- 使用 full frozen dynamics Jacobian，而非 weight matrix。
- 沿完整 ring 保存谱，不只计算单个 bump。
- slow eigenvector 与 manifold tangent 对齐，法向最大实部为负，gap 在角度上没有未解释的闭合缺口。

### Status

Implemented conditionally：accepted ring 使用 exact analytic full-dynamics Jacobian（中心差分验证）并保存 leading modes/tangent alignment；coverage 失败的 run 不计算该谱。

## Figure 5A,C: finite-time error and tangential-flow bound

### Paper meaning

训练时长附近的 angular memory error 由 slow-manifold 内最大 flow 约束；同样的 finite-time performance 可以对应不同的长期 fixed-point topology。

### Current code relation

- `learning/src/learning/analysis/phase_flow.py`
- bump/OU ensemble error metrics in `learning/src/learning/analysis/metrics.py`

### Reproduction criterion for this project

- 在同一 zero-input frozen run 中估计 `eta_theta` 与多初始角 empirical error。
- 图中明确区分 endpoint error、time-averaged error 和相应 bound。
- 覆盖率不足时不给出强结论。

### Status

Not implemented as a bound；已有所需 phase-flow 与 angular-error primitives。

## Figure 5D-E: fixed-point topology and asymptotic memory

### Paper meaning

长期误差取决于 stable fixed-point 数量和 basin 分布，而不是只取决于短时 gain。

### Current code relation

- `phase_flow.py::actual_stable_basins`
- `bump_attractor_trajectory_history.npz`

### Reproduction criterion for this project

- fixed points 由 local flow reversal 得到，并由长轨迹终点独立核验。
- 报告 basin fractions、entropy、最大 basin 宽度和 asymptotic angular error。

### Status

Partial：basin boundaries 已有，capacity/entropy 与独立终点核验尚缺。
