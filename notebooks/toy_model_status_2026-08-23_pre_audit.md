# Toy Model 研究现状与工作方向（2026-08-23，审计前草稿）

> 本文件保留 2026-08-23 阶段稿的原始判断；其中部分单位和结论已被后续审计修正。
> 当前结论请以 `toy_model_status.md` 为准。

> 本文是 `model-dev/learning` toy model 研究的阶段性总结，配合
> `slow_manifold_diagnostics.md`（诊断原理）、`idea.ipynb`（Clark 联系）、
> `sagodi.ipynb`（ghost attractor 理论）使用。会随实验进展持续更新。

---

## 1. 研究目标（终极问题）

用 **Vafidis 局部预测学习规则**（`predictive_local.py`），在 **Clark-style
heterogeneous visual cue**（GP tuning curves）的指导下训练一个环形网络，
判断学习出的 zero-input 自治动力学属于：

1. **连续吸引子流形**（切向中性、法向收缩）；
2. **离散吸引子**（$K$ 个稳定不动点）；
3. **ghost / quasi-continuous attractor**（接近连续、叠加小势垒，Sagodi 意义）。

判定标准（详见 `slow_manifold_diagnostics.md` §8）：
谱隙 $\lambda_1-\lambda_2$、切向流 $\eta=\max|\dot\theta_\parallel|$、
慢候选环覆盖、settled-fraction 分布、FP 数/势垒、时间尺度比 $R$。

## 2. 已完成实验与关键结果

### 2.1 N=60 von Mises 三 seed（seed 42/43/44，80ks，已完成 ✅）

`model-release/learning/runs/vafidis_baseline/20260822-23*`

**稳健结论（跨 seed 一致）**：
- darkness gain ≈ 0.94、OU ensemble std ≈ 2.1-2.3°、扩散指数 α ≈ 0.9-0.99
  → **速度积分是学习出来的可复现功能**；
- `dt=0.000125` 是唯一通过 numerical convergence 的步长（三 seed 一致）；
- **80ks 严重过训练**：best PI 在 16-35ks（seed 依赖），final 误差恶化到
  31-40° vs best 8-11°，stall fraction 从 ~10% 升到 50-62%，hr/hd norm 2.2→2.4；
- 学到的权重结构三 seed 一致：局部对称 + HR→HD +48° 偏移 + 正对角。

**seed 依赖结论（多稳态学习）**：
- 自治吸引子结构定性不同：seed42 → 25 个稳定 FP（干净离散环）；
  seed43 → 2 个 FP + 22 个 unresolved 区间（近连续特征）；
  seed44 → 5 个 FP + 18 个 unresolved。
- **同样的规则/参数/预算，不同初始化学到不同拓扑** —— 单 seed 结论不可靠，
  必须多 seed + checkpoint selection。

### 2.2 N=120 heterogeneous（seed 42，旧配置 50ks，已完成 ✅）

`model-dev/learning/runs/vafidis_heterogeneous/20260807-133940_...`

- **cue→release 近乎处处 identity**（slope 1.013、R² 0.993、plateau 0%）——
  heterogeneous cue 让"任意相位可初始化"；
- 但**自治流形仍是 ~20-24 个离散浅井**（间距 0.14°-64° 不均）、
  RMS displacement 33°、75 个 nonmonotonic crossings、低速度 pinning 严重
  （stall 0.42、depinning 75°/s）；
- bump 是"破的"：contrast 0.15、5-7 个局部峰、peak-vs-PVA 系统性偏 24°；
- 视觉 profile 高度异质（peak 2.0-10.6、38% 多峰、COM-peak 偏移最大 76°）。

**结论**：heterogeneous cue 改善了 cue 传递与积分器增益，但没有把离散井变成
quasi-continuous —— 学习出的环是"浅而乱的离散环"。

### 2.3 慢流形/时间尺度诊断：代码重做（2026-08-23，dev 端 ✅）

针对旧诊断的三个缺陷做了数值验证与重做（详见 `slow_manifold_diagnostics.md`）：

| 缺陷 | 修复 |
| --- | --- |
| `q` 绝对值无意义（HR 快差主导，~1e7）；慢点按相对阈值会偏向 late-time basin | `select_slow_candidate_indices` 增加 `speed_floor`（物理阈值，阈值=min(相对,floor)）+ **时间均匀子采样** |
| 0.1s 网格把"停留 0 帧"与"单帧跳变"混在 within-bin median，看起来像非保守场 | `analyze_ramesan_phase_landscape` 增加 **settled/moving 分解**（`ramesan_phase_velocity_floor`），势/根改用 moving 速度场 |
| timescale 切向阈值 10° > basin 宽度中位 ~10°，立即被越过 | `timescale_separation_tangential_threshold_deg` 10→3° |

实测验证（N=120 trained weights）：leading Jacobian 谱未被 HR 快模污染
（$\lambda_1\approx-1.8\sim-10$/s）；q 的绝对尺度确由 HD→HR 低通快差主导；
settled/moving 分解在真实数据上工作正常（E2E 通过）。

> ⚠️ 这两组诊断在实验配置中**仍默认关闭**，待新训练完成后按 §4 启用。

### 2.4 视觉衰减（"夜视"）协议：接口已实现（2026-08-23，dev 端 ✅）

核心动机（数值验证）：把训练好的权重固定、只把视觉幅值从 4 降到 ≤2，
cue 后 bump 直接塌陷（pva_strength 0.957→0.003），1s darkness 后完全崩溃 ——
**学到的递归权重只对训练时见过的 cue 强度自洽**。

- `visual.training_amplitude_schedule`（分段常数调度，类比 OU std schedule）；
- `ScheduledVisualAmplitude` + `step_vafidis_toy` 的 `visual_amplitude` 覆盖参数；
- `--init-weights <npz>`：从任意 run 的权重恢复继续训练（塑性矩阵+静态几何，
  动态状态重置可复现）；
- 协议配置 `configs/protocols/visual_anneal_vafidis.yaml`：N=60、block_multirate、
  16ks + checkpoint selection、调度 4.0→1.5→0.5；
- 测试：`test_visual_annealing.py` 4 个用例，全 dev 套件 200 passed。

## 3. 基础设施现状

### 3.1 多 release-root 快照（2026-08-23）

解决"dev 不适合长训练 + model-release 被占用"的两难：

- `promote_release.py --release-root <目录>`：每个快照独立 manifest + 训练锁；
- 已建 **`model-release-anneal/`** 快照（含 visual-anneal 协议）+ 独立环境
  **`pro-anneal`**（`learning.__file__` 指向快照），smoke run 已通过；
- 之后可按实验系列建更多快照（如 `model-release-n120`）。

### 3.2 仓库 git 状态（重要事实）

- 根 `.gitignore` 含 `/model-dev/` → **`model-dev/` 从未被 git 跟踪**
  （可追溯性依赖 promote 的 manifest 文件哈希）；`model-release/` 被跟踪
  （228 文件，当前 58 改 + 14 未跟踪，含手改 config 未与 dev 对齐）；
- 建议：尽快决定是否把 `model-dev` 纳入版本控制。

## 4. 运行中 / 待完成实验

| 实验 | 环境/位置 | 状态（16:10） | 说明 |
| --- | --- | --- | --- |
| N=60 三 seed | `pro` / `model-release` | ✅ 已完成 | §2.1 |
| N=120 heterogeneous (新配置 dt=0.000125, 80ks) | `pro` / `model-release` | 🔄 训练中 | 旧结果用 dt=0.00025/50ks，需重跑 |
| N=120 von Mises (seed 42) | `pro` / `model-release` | 🔄 训练中 | 缺失的 matched 对照 |
| visual anneal (seed 42) | `pro-anneal` / `model-release-anneal` | 🔄 训练中 | §2.4 |
| N=120 多 seed (43/44) | 待启动 | ⬜ | N=60 多稳态已证明单 seed 不可靠 |
| slow-manifold/timescale 诊断启用 | 待新训练完成 | ⬜ | 见 §5 |

## 5. 下一步工作方向（按优先级）

### P0：等训练完成后立即做（数据已就绪或即将就绪）

1. **解读 N=120 von-Mises vs heterogeneous 的 matched 对比**（新配置下）：
   cue 类型对 cue-transfer、积分器、自治流形的影响 —— 这是
   "heterogeneous 有效吗" 的可信答案；
2. **visual-anneal 结果解读**：对比 anneal vs baseline 的 darkness 性能
   （bump maintenance、低速 PI、pinning、FP 数），验证"夜视"假设；
3. **启用重做后的 slow-manifold / timescale 诊断**，对比
   `eta_theta_deg_s` / `spectral_gap_min` / `settled_fraction_median` /
   `basin_entropy` / FP 数 —— 直接回答 quasi-continuous 问题。

### P1：补齐统计可信度

4. **N=120 补 seed 43/44**（von Mises 与 heterogeneous 都补），统一
   dt=0.000125 + 16-20ks + checkpoint selection（N=60 三 seed 已证明
   80ks 过训练、多稳态）；
5. 所有结论用 `best_weights.npz`，**绝不用 final**（现行诊断已默认 best）。

### P2：科学干预（如果 heterogeneous 单独不够）

6. **视觉衰减 + checkpoint selection 正式跑**（N=60 与 N=120 各一组），
   观察是否让 settled 分布更均匀、pinning 降低；
7. **OU std 调制**（`pi_robust_vafidis.yaml` 已有 broad-to-low 调度）：
   注意不同速度要求不同 `w_hr_to_hd` 增益，存在任务间干扰风险 —— 需配
   "冻结评估低速保持率"（可用现有 `weight_snapshot_pi_development`）；
   "多速度共存下低速性能保持"本身是 quasi-continuous 的实验判据；
8. **Clark 投影分析**：把学到的 $W_{HD\to HD}$ 投影到 GP tuning 基上，
   检查平移不变性（restoring condition）是否近似成立，估计需要多大 N。

### P3：理论桥梁（回答"能不能像 Clark 一样恢复等效连续吸引子"）

9. DMFT-lite / 大 N 外推：用学到的 W + heterogeneous profiles 估计
   roughness 的 $1/\sqrt{N}$ 标度，判断小模型能否绕过 Clark 的 $N>10^4$；
10. Sagodi 全状态慢流形分析（完整 600 维 Jacobian 谱 + 有限时间记忆界），
    目前只有 cue-ring 探针，未做完整流形拟合。

## 6. 需要警惕的结论陷阱

- **N 与 cue 类型混淆**：旧 N=60 von Mises vs N=120 heterogeneous 不可直接比；
- **单 seed 不可靠**：多稳态学习已证实；
- **"cue transfer 处处 identity" ≠ 连续流形**：只说明可初始化性；
- **final 权重不可用**：过训练严重，必须 best/checkpoint；
- **"低速性能保持"才是连续吸引子的判据**，不是高速 gain≈1。

---

*下次更新：N=120 两组 + visual-anneal 训练完成后，补 P0 的对比结果。*
