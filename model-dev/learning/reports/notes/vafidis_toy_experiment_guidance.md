# Vafidis Toy Model 实验指导

## 当前结论

当前坏图不主要是因为训练时间不够长。诊断 sweep 显示：

- zero-init 条件下，训练从 20s 增加到 160s，`velocity_gain` 仍接近 0；
- `w_hd_to_hd` 的 symmetry score 会随训练时长略升，但 bump drift 没有明显改善；
- `w_hr_to_hd` 可以出现左右不对称偏移，但这个偏移目前没有转化成 darkness 中的 bump velocity；
- local-kernel recurrent warm start 能显著改善 HD recurrent structure 和部分 bump stability，但仍不能解决 velocity gain。

因此当前瓶颈更可能是模型动力学 / 测试协议 / HR-to-HD 推动机制，而不只是训练轮数。

## 推荐实验顺序

### 1. 先做 frozen-dynamics sanity check

目的：验证在完全手工权重下，动力学本身能否维持和推动 bump。

建议条件：

- `w_hd_to_hd_mode: local_kernel`
- `eta_hd_to_hd: 0`
- 手工或初始化 shifted `w_lhr_to_hd` 与 `w_rhr_to_hd`
- testing phase 完全冻结权重

若手工 shifted HR-to-HD 都不能让 bump 在 darkness 中移动，则学习规则不是首要问题，应先改动力学或 HR 输入解释。

### 2. 再做 recurrent-only 学习

目的：只回答 HD bump maintenance 是否能学出来。

建议条件：

- `velocity.value: 0`
- `eta_hr_to_hd: 0`
- training phase 使用 visual teacher
- 测试 bump maintenance，不测 path integration

成功标志：

- `w_hd_to_hd` 接近局部对称 kernel；
- bump test 中 `bump_final_abs_drift` 明显小于 0.5 rad；
- activity heatmap 中 bump 不塌成均匀背景，也不锁死到固定非 cue 方向。

### 3. 再做 HR-to-HD 学习

目的：在已能维持 bump 的 recurrent backbone 上学习速度驱动。

建议条件：

- 使用 fixed 或 warm-start `w_hd_to_hd`
- `eta_hd_to_hd: 0`
- 开启 `eta_hr_to_hd`
- 系统扫描 `tau_delta`、`eta_hr_to_hd`、`k_vel`

成功标志：

- `w_lhr_to_hd` 与 `w_rhr_to_hd` 出现相反方向偏移；
- darkness 中 decoded heading 随 angular velocity 同方向移动；
- velocity gain curve 斜率非零，并随 `k_vel` 单调变化。

### 4. 最后合并联合训练

只有当前两项分别成立后，再同时更新 `w_hd_to_hd` 和 `w_hr_to_hd`。否则联合训练得到坏图时很难判断是 recurrent、HR drive，还是测试协议出了问题。

## 本轮 sweep 结果摘要

结果文件：

```text
learning/reports/notes/vafidis_toy_diagnostic_sweep.csv
learning/reports/notes/vafidis_toy_diagnostic_sweep.json
```

关键观察：

| condition | bump drift | final PI error | velocity gain | HD symmetry |
| --- | ---: | ---: | ---: | ---: |
| zero 20s | 0.599 | 2.166 | ~0 | 0.946 |
| zero 40s | 0.606 | 2.158 | ~0 | 0.948 |
| zero 80s | 0.616 | 2.147 | ~0 | 0.950 |
| zero 160s | 0.633 | 2.131 | ~0 | 0.952 |
| local-kernel 40s | 0.322 | 1.359 | ~0 | 0.998 |
| strong-HR 80s | 0.373 | 2.325 | ~0 | 0.942 |

解释：

- 长训练只让权重指标缓慢变化，没有打开 path integration；
- local-kernel backbone 对 bump 和 HD weight structure 有帮助；
- HR 学习强度变大并不会自动产生 velocity gain，说明 HR-to-HD 偏移结构还没有成为有效的 travelling-wave drive。

## 追加尝试：hand-shifted HR baseline

追加运行：

```text
learning/reports/notes/hand_shifted_hr_diagnostic.csv
learning/reports/notes/hand_shifted_hr_diagnostic_wide.csv
```

做法：

- 使用 `recurrent_only_probe` 中已经能稳定维持 bump 的 recurrent backbone；
- 直接用手工 shifted Gaussian kernel 替换 `w_lhr_to_hd` 和 `w_rhr_to_hd`；
- 扫描 shift = 0.15, 0.30, 0.50, 0.80 rad；
- 扫描 scale = 0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 3.00。

结果：

- scale <= 0.80 时 bump maintenance 很稳定，但 `velocity_gain` 仍约等于 0；
- scale >= 1.50 时部分条件下 PVA 变 NaN，说明活动趋于饱和或失去可解码 bump；
- 因此，即使用手工 HR asymmetric weight，当前动力学也没有把 HR current 转换成连续 bump 运动。

这进一步说明：不要继续单纯拉长训练。下一步应先修正或扩展测试期动力学，使手工 HR baseline 能产生非零 bump velocity，然后再回到 local learning。

## 下一步优先任务

1. 加一个 hand-designed shifted HR-to-HD baseline。
2. 给 activity/current traces 增加诊断图：`r_lhr`、`r_rhr`、`i_hd_distal`、`v_hd_ss`。
3. 如果 hand-designed baseline 仍不能移动 bump，优先修改动力学，例如加入更明确的 divisive normalization、Mexican-hat recurrent backbone 或速度依赖 asymmetric current。
4. 当前已新增 recurrent-only / HR-only / long-train config，后续应围绕这三类 config 做对照。
