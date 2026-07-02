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
