# 2026-08-24 N=60 三 seed 重跑核对 + slow_manifold IndexError 修复

## 1. 发现的 bug（已修复，dev+release 同步）

**症状**：N=60 三个 seed 用 `vafidis_diagnostics_n60.yaml` 重跑诊断时，
`bump_attractor_trajectories` 报 `IndexError: index 31 out of bounds for axis 0
with size 30`，导致：
- `bump_attractor_trajectories` 失败 → 无新的 trajectory（含 autonomous state）；
- `slow_manifold` 因 `ramesan_missing_probe_state=1` 退化失败；
- `timescale_separation` 报 "requires enabled bump attractor trajectories" 失败。

**根因**：`slow_manifold.py::select_slow_candidate_indices` 的时间均匀子采样分支，
`selection[filled:] = candidate_index[remainder]` 把**候选数组的值**当成了
**候选下标**存入 `selection`，随后 `candidate_index[selection]` 越界。触发条件：
N=60 轨迹 51 采样点、慢候选 ~30 个、预算 12（`ceil(4096/360)=12`）。

**修复**：`selection` 统一存候选位置（`slot_indices[0]`），末尾统一
`candidate_index[selection]` 映射一次。新增回归测试
`test_slow_candidate_time_resampling_fills_sparse_slots`。dev 201 passed，
已 promote 到 release。

**影响**：修复前 N=60 三 seed 的 slow_manifold / timescale / endpoint 指标全部
无效（nan 或退化），需重跑；N=120 候选多不受影响，但建议修复后重跑以统一。

## 2. N=60 三 seed 可靠行为结论（修复前已有，不受 bug 影响）

（指标来自 `runs/vafidis_baseline/20260822-*`，`weight_source: best`）

| 指标 | seed42 | seed43 | seed44 | 解读 |
| --- | --- | --- | --- | --- |
| darkness velocity gain | 0.938 | 0.936 | 0.935 | 三 seed 一致，强 cue + 长训练学到 gain≈0.94 |
| gain abs error | 0.062 | 0.064 | 0.065 | 稳定 |
| constant PI final RMS error | 8.36° | 8.49° | 6.91° | 中高速积分误差 ~7-8° |
| zero-velocity drift | 0 | 0 | 0 | bump 维持完美 |
| OU ensemble mean±std | 0.70°±2.30 | 0.26°±2.11 | 3.20°±2.22 | seed44 系统性偏 3° |
| OU effective diffusion | 290 | 243 | 269 °²/s | 与扩散一致 |
| OU anomalous exponent | 0.989 | 0.909 | 0.903 | 接近正常扩散(α≈1) |
| best snapshot time | 35.2 ks | 20.8 ks | 16.0 ks | 过训练普遍，best 在 16-35 ks |
| best time-avg PI error | 9.3° | 8.1° | 11.2° | best 权重下 |
| depinning velocity | 30 | 30 | 30 °/s | 低速 pinning 需 30°/s 才解 |
| best stall fraction | 0.21 | 0.13 | 0.04 | seed44 最好 |
| hd-to-hd symmetry | 0.976 | 0.956 | 0.957 | 高度局部对称 |
| LHR/RHR→HD offset | −1.26/+1.26 | −1.24/+1.27 | −1.23/+1.26 | 左右镜像、与论文一致 |
| hr/hd norm ratio | 2.40 | 2.39 | 2.38 | 一致 |

**稳健结论**（三 seed 一致）：
1. **局部预测学习规则可学习 bump 维持**：zero-drift=0、PVA strength≈0.95、
   contrast 0.148，且权重高度局部对称（symmetry≈0.96）——与 Vafidis 论文
   HD→HD 局部对称结构一致；
2. **HR→HD 学到左右镜像偏移**（LHR 负 / RHR 正偏移，±1.26 rad），与论文
   Fig.3C 的 asymmetric 结构方向一致，但**主连接符号为负**（之前 audit 发现，
   与论文正主连接不同）——这是 self-consistent 但非论文的解；
3. **gain≈0.94 但低速 pinning**：depinning 需 30°/s、stall 4-21%。即
   "速度积分可用但不连续"——与离散吸引子预期一致；
4. **过训练普遍**：best 在 16-35 ks，final 显著更差（之前 baseline 分析）。

## 3. 修复后慢流形诊断（2026-08-24 重跑）—— 科学结论

修复后 N=60 seed42 慢流形诊断能完整跑通，**但 fit 仍失败，原因是
`slow_manifold_fit_failure_is_insufficient_coverage=1`**：

- `slow_manifold_angular_support_fraction = 0.1`（候选只覆盖 ~10% 的角度 bin）；
- 候选点聚成少数角度簇（小规模 36 初始条件下 12 个簇 / 36 bins；正式 360 IC 下
  同样 coverage 不足）；
- `ramesan_diagnostic_succeeded=1`，phase landscape 给出 15 stable + 14 unstable
  交替的 fixed points（离散环特征）；
- `slow_manifold_candidate_speed_median ≈ 9e-5`（候选确实接近不动点）。

**这是科学结论而非 bug**：N=60 学出的网络，低速度候选状态（接近不动点的状态）
只出现在少数 heading 附近、不铺满整个环 —— 正是 **离散 phase-locking / discrete
attractor** 的诊断特征，与论文"有限 N 下存在 discrete attractors/basins"的
描述定性一致。`min_angular_support_fraction=0.5` 门槛对离散网络必然失败，
这是诊断设计使然（它本就要区分连续 vs 离散）。

**组会可用措辞**：N=60 的慢候选呈 ~30 个离散相位锁定（phase landscape 显示
stable/unstable 交替），不是覆盖全环的连续慢流形。

## 4. 待办
- [ ] N=60 三 seed 用修复后 release 重跑 `vafidis_diagnostics_n60.yaml`
      （轨迹 + 慢流形 + timescale + endpoint 全开）—— 确认 43/44 的 coverage；
- [ ] N=120 两 run 当前诊断结束后，用修复后代码重跑 `n120` 诊断；
- [ ] 汇总 FP 数 / 谱隙 / eta / settled-fraction，判断吸引子类型；
- [ ] 组会图像准备（见 toy_model_status.md 更新）。
