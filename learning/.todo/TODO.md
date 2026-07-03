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
