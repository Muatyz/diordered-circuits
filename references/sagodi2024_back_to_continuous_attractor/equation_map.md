# Equation Map: sagodi2024_back_to_continuous_attractor

## Eq. 2-4: perturbed fast-slow dynamics

### Paper form

`x_dot = f(x) + epsilon p(x)`，在流形附近写成切向 `y_dot = epsilon g(y,z,epsilon)` 与法向 `z_dot = h(y,z,epsilon)`。

### Discrete implementation

- 先定义冻结权重、零视觉、零速度、零噪声的完整一步映射 `G_dt(state)`。
- 对 Euler baseline，连续流近似为 `F(state) = (G_dt(state) - state) / dt`。
- 对离散一步 Jacobian `J_G`，对应 flow Jacobian 为 `(J_G - I) / dt`；报告 time constant 时必须使用 flow eigenvalue，而不是直接使用 `J_G`。
- state 必须包含会影响下一步的全部动态量。按当前实现，至少要审计 `r_hd_to_hr_lp`、`r_hr`、`i_hd_distal`、`v_hd_distal`；权重和 plasticity traces 在 frozen analysis 中是参数而不是 state。

### Code target

- Planned file: `learning/src/learning/dynamics/autonomous.py`
- Planned interfaces: `pack_autonomous_state`, `unpack_autonomous_state`, `autonomous_step`, `autonomous_flow`
- Existing source step: `learning/src/learning/models/vafidis_toy.py::step_vafidis_toy`

### Numerical risks

- 当前一步更新中 HR 与 HD pathway 存在显式 lag；漏掉 `r_hr` 会得到错误 Jacobian。
- derived fields、teacher angle、PSP traces 和 weights 不应重复进入 frozen state vector。
- finite-difference step 需按每类变量的典型尺度设定，不能只用一个绝对 epsilon。

### Validation

- `autonomous_step` 应与 `step_vafidis_toy(..., angular_velocity=0, visual_teacher=False, training=False)` 一步结果一致。
- directional finite difference 应与 Jacobian-vector product 一致。
- `dt` 减半时主要 flow eigenvalues 应收敛。

## Theorem 1 / Proposition 1: persistent slow manifold and distance to a continuous attractor

### Paper form

紧致、连通、normally hyperbolic 的 continuous attractor 在小扰动下保留同胚 invariant slow manifold；若流形切向 flow 的 uniform norm 为 `eta`，则存在不超过 `eta` 的 vector-field perturbation 可将其变成 continuous attractor。

### Discrete implementation

- 从多初始角的 autonomous darkness trajectories 中提取慢速尾段。
- 按 decoded angle 排序并拟合 periodic curve，保留 full-state support points 与输出角坐标。
- 计算每个 support point 的 full-state flow、切向投影、法向残差和 `eta = sup |theta_dot|`。
- 独立扰动 full state 的法向方向，验证轨迹回到该 autonomous curve。

### Code target

- Planned file: `learning/src/learning/analysis/slow_manifold.py`
- Existing partial probes: `run_bump_attractor_trajectory_test`, `run_timescale_separation_test`, `learning/analysis/phase_flow.py`

### Numerical risks

- visual-teacher tuning curve 不一定 invariant，不能替代 autonomous manifold。
- 只在 `v_hd_distal` 子空间算距离可能漏掉 HR/current 方向的偏离。
- periodic spline 可能跨越稀疏采样区；必须保存 angular coverage 与相邻点距离。

### Validation

- ring closure error、angular coverage、nearest-manifold reconstruction error。
- autonomous one-step normal residual应显著小于随机 state-cloud baseline。
- state perturbation 后 full-state distance 应快速衰减，而 decoded angle 的变化慢得多。

## Eq. 5 / Eq. 41: finite-time memory-error bound

### Paper form

平均记忆偏移满足 `mean_M |x(t,x0)-x0| <= t ||phi||_infinity`。

### Ring/output implementation

- 使用 PVA/COM 输出角，定义 circular error。
- `eta_theta = max_theta |F_0(theta)|`，单位 rad/s，其中 `F_0` 为 zero-input phase flow。
- 对每个 elapsed time 保存 `mean_abs_circular_error(t)` 与 conservative bound `t * eta_theta`（截断到 `pi`）。
- 若报告从 `0..T` 的时间平均误差，对近似常速漂移可同时展示 `T * eta_theta / 2`；不要与 endpoint bound 混用。

### Code target

- Existing: `learning/src/learning/analysis/phase_flow.py`
- Existing metric helpers: `learning/src/learning/analysis/metrics.py`
- Planned output: `slow_manifold_summary.npz/json`, `figures/heading/finite_horizon_memory_bound.png`

### Validation

- 每个时间点检查 empirical mean error 不超过注明定义的 bound；若失败，优先检查 phase-flow coverage、decoder singularity 和 manifold identification。

## Eq. 67-68: fixed-point basins and asymptotic memory capacity

### Paper form

stable fixed point 的 basin 由沿一维 manifold 的 flow direction 决定；均匀角先验下，asymptotic memory capacity 与 basin fractions 的熵相关。

### Discrete implementation

- 使用 `phase_flow.py` 的稳定/不稳定根及相邻 basin boundaries。
- 对每个 stable basin 计算 circular arc fraction `p_k`。
- 保存 `basin_entropy = -sum p_k log(p_k)`；论文写作使用 negative conditional entropy 的符号约定，代码字段名应避免符号歧义。
- 同时保存 stable/saddle count、max basin width、mean inter-fixed-point distance 与 worst asymptotic angular error。

### Code target

- Extend: `learning/src/learning/analysis/phase_flow.py`
- Tests: `learning/tests/test_phase_flow.py`

### Numerical risks

- scalar phase-flow root 必须交替 stable/saddle；不交替通常说明 smoothing、覆盖或 decoder 有问题。
- fixed-point count 对 smoothing 和 bin size 敏感，必须连同配置保存。

## S7.6.3: Jacobian eigenspectrum along the manifold

### Paper criterion

一维 attractive slow manifold 应有一个最接近零的 slow mode；第二大及其余 eigenvalue 的实部应更负，并且这种 gap 要沿整条 manifold 一致存在。

### Discrete implementation

- 在每个 autonomous manifold anchor 上计算 full-state flow Jacobian。
- 保存所有 eigenvalues、slow eigenvector 与 manifold tangent 的 alignment。
- 推荐指标：`slow_rate = max Re(lambda)`、`fast_edge = second largest Re(lambda)`、`spectral_gap = slow_rate - fast_edge`、`normal_margin = -fast_edge`。
- 仅当所有法向 modes 收缩且最慢 eigenvector 与几何 tangent 对齐时，才将结果解释为 normal hyperbolicity evidence。

### Code target

- Planned file: `learning/src/learning/analysis/jacobian.py`
- Do not substitute: `learning/src/learning/analysis/weights.py::compute_weight_eigenvalues`

