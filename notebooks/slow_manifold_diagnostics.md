# 慢流形诊断：实验设计与数学原理

> 中文查阅笔记 · 对应代码：`model-dev/learning/src/learning/analysis/slow_manifold.py`、
> `model-dev/learning/src/learning/experiments/run_vafidis_toy.py`（`run_slow_manifold_diagnostic` /
> `run_timescale_separation_test`）
>
> 配套理论笔记：`sagodi.ipynb`（ghost attractor / persistent manifold theorem）、
> `idea.ipynb`（Clark DMFT 与小网络的联系）、`Vafidis.md`（slow-manifold state 与 PCA）。
> 2026-08-23 完成代码修订（物理阈值、时间均匀子采样、settled/moving 分解），本文与之同步。

---

## 1. 我们要回答的科学问题

Vafidis 局部预测学习规则 + Clark-style heterogeneous visual cue 训练出的网络，
其 **zero-input（darkness、v=0）自治动力学** 到底是哪一种：

1. **连续吸引子流形**：存在一维流形 $\mathcal{M}\subset\mathbb{R}^D$，其上每一点都是
   不动点（切向中性、法向收缩）；
2. **离散吸引子**：只有 $K$ 个孤立稳定不动点（势阱），周围是 basin；
3. **ghost / quasi-continuous attractor**：接近连续流形、但叠加了小势垒
   （Sagodi 意义下"连续吸引子的幽灵"）。

三种情形在行为上（bump 维持、路径积分）很难只靠 endpoint map 或 gain 曲线区分，
因为它们都"看起来能维持 heading"。慢流形诊断测的是 **状态空间动力学的几何**，
而不是某个解码器的时间序列。

---

## 2. 理论框架：Sagodi 的快慢分解

记动力学为

$$
\dot{x} = f(x), \qquad x\in\mathbb{R}^{D}.
$$

若存在 $l$ 维连续吸引子 $\mathcal{M}_0$，则

$$
f(x) = 0 \quad \forall x\in\mathcal{M}_0,
$$

且沿流形的切向 Jacobian 本征值为零、法向本征值实部严格为负：

$$
\lambda_{\parallel} = 0, \qquad \operatorname{Re}(\lambda_{\perp}) < -\delta < 0.
$$

任意小扰动 $f \to f + \epsilon p$ 会把 $\lambda_\parallel=0$ 分裂为 $\lambda_\parallel=\pm\epsilon$。
把状态分解为 $x = (q_\parallel, q_\perp)$（切向/法向坐标），得到快慢系统

$$
\begin{cases}
\dot{q}_\parallel = \epsilon\, g(q_\parallel, q_\perp, \epsilon), \\[2mm]
\dot{q}_\perp = h(q_\parallel, q_\perp, \epsilon),
\end{cases}
\qquad
h(q_\parallel, q_\perp, 0) = 0 \;\Rightarrow\; \mathcal{M}_0.
$$

**持久流形定理（persistent manifold theorem）**：只要法向双曲性
$\operatorname{Re}\lambda_\perp < -\delta$ 一致成立，扰动后的系统仍存在一个
**慢流形** $\mathcal{M}_\epsilon = \{(q_\parallel, q_\perp): q_\perp = c_\epsilon(q_\parallel)\}$，
其上动力学为

$$
\dot{q}_\parallel = \epsilon\, g(q_\parallel, c_\epsilon(q_\parallel), \epsilon),
\qquad \text{切向速度 } \mathcal{O}(\epsilon).
$$

**记忆误差上界**：流形上的切向速度场记 $\phi = f|_{\mathcal{M}_\epsilon}$，则

$$
e(T) = \|x(T) - x_0\| \le \int_0^T \|\phi(x(s))\|\,ds \le T\,\|\phi\|_\infty.
$$

只要 $T\|\phi\|_\infty \ll \Delta$（行为容差），慢流形在功能上与真正的连续吸引子
无法区分。长期来看，系统受慢流形上的不动点或极限环控制：

- **$K$ 个均匀稳定不动点**：最大长期误差 $e_{\max}\approx \pi/K$；
- **极限环**：角度持续漂移，渐近误差趋于 $\pi$（完全失去记忆）。

> **判据即测量**：谱隙（§5.1）、切向流 $\|\phi\|_\infty$（§5.2）、慢候选的环覆盖
> （§5.3）、FP 数/分布/势垒（§5.4）正是上面几个量的可计算版本。

---

## 3. 最小 Markov 状态与离散映射

冻结权重、关闭视觉、速度为零时，Vafidis toy 的最小 Markov 状态为

$$
x = [\, r_{HD\to HR}^{LP},\; r_{HR},\; i_{HD,d},\; v_{HD,d},\; v_{HD,a}\,]
\;\in\;\mathbb{R}^{D}, \qquad D = 4N_{HD} + N_{HR}.
$$

（$N_{HD}=N_{HR}=120$ 时 $D=600$。）这五个分块各自是：HD→HR 低通率、一步滞后的
HR 发放率、HD distal 电流、distal 电压、proximal 电压。它们构成完整的自洽闭环
（`FrozenAutonomousDynamics`，`learning/dynamics/autonomous.py`）。

数值上我们只有离散映射

$$
x_{n+1} = G_{\Delta t}(x_n),
$$

于是定义 **Euler 等价流** 与 **q 函数**：

$$
F_{\Delta t}(x) = \frac{G_{\Delta t}(x) - x}{\Delta t},
\qquad
q(x) = \frac{1}{2}\,\bigl\|F_{\Delta t}(x)\bigr\|^2.
$$

$q$ 小 $\Leftrightarrow$ 接近不动点。**注意**：$q$ 的绝对值依赖状态单位
（rad/s、kHz 混在一起），实测在 cue-release 态 $q\sim 10^7$，主要由 HD→HR 低通
快差（$\|\dot r_{HD\to HR}\|\sim 7\times10^3$/s）贡献，**不是** HR 代数快模。因此
"$q$ 小于某个固定阈值"作为慢点判据没有意义——必须用每条轨迹的相对/物理阈值
（见 §5.3）。

Jacobian 在完整状态上解析计算（`flow_jacobian`），不在 PCA 投影上做。

---

## 4. 实验协议：cue → release → darkness

所有慢流形诊断共用同一套冻结权重协议（`run_bump_attractor_trajectory_test`）：

1. **cue（视觉锚定）**：从均匀角度网格（默认 360 个初始角度 $\theta_{\text{init}}$）
   出发，施加诊断用强 cue（`bump_attractor_cue_amplitude=24`，比训练 cue 强，
   使每个相位都能初始化），时长 1 s；
2. **release**：记录 cue-off 时的完整状态 $x_0$（`autonomous_probe_state`）；
3. **darkness（自治演化）**：冻结权重、无视觉、$v=0$，演化 $T=5$ s，
   按 `sample_interval=0.1$s` 记录轨迹 $x(t)$、解码相位 $\theta(t)$ 与
   $\|F_{\Delta t}(x)\|$。

这套协议保证：**视觉只负责把网络放到环上某点，自治动力学随后完全由学到的权重
决定**。cue 传递的忠实性由 endpoint map 单独诊断（`trajectory_and_fixed_points`
组），不混入这里的几何测量。

---

## 5. 四个判据

### 5.1 Jacobian 谱隙（判据 A：法向收缩 + 切向中性）

在慢候选点 $x$ 处计算 $J(x) = \partial F/\partial x$ 的 leading 本征值：

$$
\operatorname{Re}\lambda_1(x) \ge \operatorname{Re}\lambda_2(x) \ge \cdots
$$

定义 **谱隙**

$$
\Delta\lambda(x) = \lambda_1(x) - \lambda_2(x),
$$

与 **leading 模的切向对齐度**

$$
a(x) = \frac{\bigl|\,t(\theta) \cdot \hat v_1(x)\,\bigr|}
{\|t(\theta)\|\,\|\hat v_1(x)\|},
$$

其中 $t(\theta)$ 是慢流形样条在 $\theta$ 处的切向，$\hat v_1$ 是 $\lambda_1$ 对应的
特征向量。

| 情形 | $\lambda_1$ | 谱隙 | $a$（leading 模 vs 切向） |
| --- | --- | --- | --- |
| 连续吸引子 | $\approx 0$（处处） | 大且稳定 | 接近 1 |
| 离散 FP | $<0$，随 $\theta$ 振荡 | 小/在 FP 处消失 | 小（leading 模法向） |
| ghost | 接近 0、缓慢变化 | 中等 | 中等 |

实测（N=120 heterogeneous）：cue-release 态 $\lambda_1\approx -1.8\sim-10$/s、
$\lambda_3\approx-12$/s——leading 谱是 HD-distal 时标，**未被 HR 快模
（$-1/\Delta t=-4000$/s）污染**，谱是可信的。

### 5.2 切向流 $\eta$ 与记忆误差上界（判据 B：流形平坦度）

慢候选点用周期样条拟合流形 $x = c(\theta)$（`fit_periodic_state_curve`），
切向量 $t(\theta) = c'(\theta)$。沿流形的 **切向相位流** 为

$$
\dot\theta_\parallel(\theta) = \frac{t(\theta)\cdot F_{\Delta t}(c(\theta))}
{\|t(\theta)\|^2},
$$

**法向流** 为 $F - t\,\dot\theta_\parallel$。定义

$$
\eta = \max_\theta \bigl|\dot\theta_\parallel(\theta)\bigr|
\;\;[\text{rad/s}]
$$

这是 Sagodi 记忆误差上界 $e(T)\le T\|\phi\|_\infty$ 在相位坐标下的直接来源
（`slow_manifold_eta_theta_deg_s`）。若 $\eta$ 对应的 $T\eta$ 远小于行为容差，
网络在 $T$ 时间内与连续吸引子功能等价。

### 5.3 慢候选集与环覆盖（判据 C：吸引子是否铺满整个环）

候选点按**每条轨迹**的相对速度阈值 + **物理速度下限**筛选：

$$
\|F_{\Delta t}(x)\| \le \min\Bigl(\epsilon \cdot \max_t \|F_{\Delta t}(x(t))\|,\;
v_{\text{floor}}^{(s)}\Bigr),
\qquad \epsilon = 10^{-3},\;
v_{\text{floor}}^{(s)} = 5\times10^{-4}\,\text{rad/s},
$$

超预算时按**时间均匀**重采样（不是按索引），避免候选点全部挤在 late-time basin。

> **为什么需要物理下限**：轨迹的 $\max\|F\|$ 由 cue-off 的弛豫瞬态决定
> （$\sim 7\times10^3$/s），单靠 $\epsilon\cdot\max$ 得到的阈值 $\sim7$/s 会把
> 弛豫中期的点也算作"慢"。物理下限把候选限制在真正接近吸引子的状态。

候选点按角度分箱后，`angular_support_fraction` = 被候选覆盖的角度 bin 比例：

- **覆盖整个环**（support ≈ 1）→ 吸引子连续/致密 → 支持 quasi-continuous；
- **聚成若干孤立簇**（support 明显 < 1，且簇数与 FP 数一致）→ 离散吸引子。

> 注意：候选覆盖出现角度缺口时，**先检查 darkness 时长是否足够**。近 saddle-node
> 对（临界慢化）轨迹在有限时长内达不到物理下限，缺口本身有信息量（慢化区），
> 但不要误判为"流形不完整"。

### 5.4 有效势与不动点结构（判据 D：势垒高度与 FP 密度）

在相位网格上，一维相位流可写成保守势 + 恒定漂移的形式：

$$
v(\theta) = \bar v - \frac{dU(\theta)}{d\theta}.
$$

$U$ 在 Fourier 域积分（`_periodic_effective_potential`）：

$$
\tilde U(\omega) = -\frac{\tilde v_{ac}(\omega)}{i\omega},
\qquad U(\theta) = \mathcal{F}^{-1}\bigl[\tilde U\bigr] - \min U.
$$

其中 $v_{ac} = v - \bar v$，$\bar v\neq 0$ 表示非保守环流（极限环的征兆）。
势垒高度 $\max U - \min U$ 直接对应离散井的深度。

不动点是 $v(\theta)=0$ 的根，稳定/不稳定交替出现（$\dot\theta$ 过零的斜率符号）。
由它们得到：

- **稳定 FP 数 $K$** 与最大间距 → 长期误差 $e_{\max}\approx \pi/K$；
- **basin 宽度分布** → **basin 熵** $H = -\sum_i w_i\ln w_i$（$w_i$ 为各 basin 占
  环的比例）：均匀环 → $H\to\ln K$（最大）；若干大 basin + 若干小 basin → $H$ 小。

---

## 6. 采样伪影与 settled/moving 分解

### 6.1 问题

相位速度在 0.1 s 网格上估算：

$$
v_i = \frac{\operatorname{wrap}\bigl(\theta_{i+1} - \theta_i\bigr)}{\Delta t_s},
\qquad \Delta t_s = 0.1\,\text{s}.
$$

bump 停在不动点时，连续多帧 $v_i=0$；一旦移动，在单帧内跳过一个 bin，
$|v_i|$ 瞬间变得很大。把两者混在 within-bin median 里，IQR 同时含 0 与跳变率，
看起来像一个"非保守、噪声巨大的速度场"——这是采样伪影，不是动力学性质。

### 6.2 修复：settled / moving 分解

按物理速度下限 $v_{\text{floor}}=10^{-3}$ rad/s（≈0.06°/s，高于 FP 处 PVA 数值抖动、
低于真实漂移）把帧分为两类：

$$
\text{settled: } |v_i| < v_{\text{floor}}, \qquad
\text{moving: } |v_i| \ge v_{\text{floor}}.
$$

- **settled fraction**（bin 内停留时间占比）$\in[0,1]$：吸引子 bin ≈ 1，
  过渡区 ≈ 0。它等价于"该相位被网络当作稳定记忆存储的时间比例"。
- **moving velocity**（仅对 moving 帧取中位数）：bump 实际跨越该相位区间的漂移率。

平滑、有效势、根查找一律使用 **moving 速度场**（settled 区隐含其零值通过
settled fraction 表达）。这样面板 B 的两个物理量各司其职：
**settled fraction = 吸引子位置；moving velocity = 吸引子间的势垒/漂移**。

### 6.3 时间均匀子采样（消除 late-time 偏置）

轨迹从 $\|F\|\sim10^3$ 指数掉到 $<10^{-9}$，5 s 内绝大多数时间停在 FP。
若按索引等间隔采样，候选几乎全来自 FP basin。改为按**时间分槽**均匀采样：
每条轨迹只保留 `slow_manifold_candidate_count / n_initial_conditions` 个点，
且这些点在时间轴上均匀分布，弛豫段与停留段按时长公平贡献。

---

## 7. Timescale separation 判据（判据 E：快慢分离）

`run_timescale_separation_test` 用操作化方式测 Clark Figure 3 风格的时间尺度分离：

### 法向弛豫时间 $\tau_\perp$

1. 对每个 cue anchor，施加 1 s 视觉 cue 得到 release 态；
2. 在 HD distal current 空间沿**环法向**加随机扰动（RMS $=0.025,\,0.05,\,0.1$，
   每条件 3 次重复；扰动同时加在 $i_{HD,d}$ 与 $v_{HD,d}$，保持 proximal 连续）；
3. 在 darkness 中测量到最近闭合流形（piecewise-linear manifold，见
   `nearest_closed_manifold_distance`）的距离 $d(t)$；
4. 从峰值到 $1/e$ 的 e-folding 时间即 $\tau_\perp$。

$$
d(t) = \min_{\text{segments}} \Bigl\| \frac{x(t) - \Pi(x(t))}{\sqrt{D'}} \Bigr\|,
\qquad
\tau_\perp :\; d(\tau_\perp) = d_{\max}/e.
$$

### 切向漂移时间 $\tau_\parallel$

用 bump_attractor 轨迹的 **Clark-overlap 位移**首达时间：从 release 起，
overlap 解码位移首次超过阈值 $\theta_{\text{thr}}$（修订后 3°，原 10°）的时间。

$$
\tau_\parallel = \min\bigl\{t : |\Delta\theta_{\text{overlap}}(t)| \ge \theta_{\text{thr}}\bigr\}.
$$

> **为什么从 10° 改到 3°**：N=120 heterogeneous 网络稳定 FP 的 basin 宽度中位
> 约 10°，10° 阈值几乎立即被越过，$\tau_\parallel$ 不再反映 basin 内漂移；3° 位于
> cue-release 对齐误差（~1°）之上、basin 宽度之下。

### 保守判据

$$
R = \frac{\tau_\parallel^{p10}}{\tau_\perp^{p90}},
$$

（切向取 10 分位、法向取 90 分位，保守估计快慢分离下限。）通过条件：
$R \ge 10$ 且法向恢复观测率 $\ge 0.90$。报告时**必须同时**给出
`tangential_passage_fraction`（多少轨迹在时长内真的越过了阈值）与原始位移轨迹，
因为如果大多数轨迹根本没越过阈值，$R$ 只是下限（`ratio_is_lower_bound=1`）。

---

## 8. 结果解读指南（如何下结论）

| 测量 | 连续吸引子 | ghost / quasi-continuous | 离散吸引子 |
| --- | --- | --- | --- |
| $\lambda_1$ 沿环 | $\approx 0$ 处处 | 接近 0、缓慢起伏 | $<0$，在 FP 处最负 |
| 谱隙 $\lambda_1-\lambda_2$ | 大且稳定 | 中等 | 小，FP 处趋于 0 |
| $\eta=\max\|\dot\theta_\parallel\|$ | $\to 0$ | 小（$T\eta\ll$ 容差） | 大（势垒决定） |
| 慢候选环覆盖 | ≈1 | 接近 1（有慢化缺口） | <1，聚成 $K$ 簇 |
| settled fraction 分布 | 均匀（无特殊相位） | 较均匀 | 集中在小而深的 bin |
| moving velocity | 处处 ≈ 0 | 处处小 | 过渡区大 |
| FP 数 $K$ / basin 熵 | 无离散 FP（或极多） | 多而密 | 少（如 24）且熵小 |
| 时间尺度比 $R$ | — | 中等 | 大（势垒高） |

**单一指标不足以定论**。例如：离散网络在高速命令下 gain≈1 并不代表连续；
而"cue transfer 处处 identity"只说明可初始化性，不说明自治流形平坦。最稳健的
做法是同时看 **谱隙沿环的最小值 + $\eta$ + 慢候选环覆盖 + settled 分布**，
并在 von-Mises 与 heterogeneous 两个 matched run 之间做**横向对比**：

> heterogeneous 若使 $\eta$ 更小、谱隙更大、settled 更均匀 → 支持
> "heterogeneous cue 使学习出的吸引子更接近连续"；否则它只是改善了 cue 传递与
> 积分器增益，自治流形仍是离散环（当前数据最可能的解读：~24 个浅而不均的井）。

---

## 9. 与相关文献/笔记的关系

| 来源 | 贡献 | 本文对应 |
| --- | --- | --- |
| **Sagodi 2024** | fast-slow 分解、persistent manifold theorem、记忆误差上界 | §2、§5.2、§5.4 |
| **Clark 2025** | 全状态 Jacobian 谱、restoring condition、Figure 3F 法向弛豫 | §5.1、§7（normal 半边） |
| **Ramesan et al.** | $q=\frac12\|F\|^2$ 慢点判据、PCA 可视化 | §3、§5.3 |
| **Vafidis 2022** | 2-compartment 学习规则与权重 | §3（状态分块） |

注意区分三件事：**PCA 只是可视化**（PC1–3 解释了 ~55% 也不影响一维闭环结论）；
**q 的绝对值不能跨网络比较**（必须用相对/物理阈值）；**decoded phase 不是状态**——
相位速度场是投影，完整状态的切向/法向流与 Jacobian 才是几何判据。

---

## 10. 代码映射速查

| 概念 | 代码 |
| --- | --- |
| 冻结自治动力学、$G_{\Delta t}$、$F$、$J$ | `learning/dynamics/autonomous.py::FrozenAutonomousDynamics`（`step`/`flow`/`flow_jacobian`） |
| q、PCA、Jacobian 谱 | `slow_manifold.py::analyze_ramesan_firing_rate_geometry` |
| 慢候选阈值 + 时间子采样 | `slow_manifold.py::select_slow_candidate_indices`（`speed_floor`/`time`） |
| 慢流形样条、切向/法向流、FP、basin | `slow_manifold.py::analyze_slow_manifold_candidates` |
| settled/moving 相位场、有效势 | `slow_manifold.py::analyze_ramesan_phase_landscape`（`phase_velocity_floor`） |
| 法向/切向时间尺度 | `run_vafidis_toy.py::run_timescale_separation_test` |
| 轨迹收集、候选、probe state | `run_vafidis_toy.py::run_bump_attractor_trajectory_test` |
| 关键配置 | `slow_manifold_speed_floor`、`ramesan_phase_velocity_floor`、`timescale_separation_tangential_threshold_deg`（见 `configs/diagnostics/vafidis_diagnostics.yaml`） |
| 绘制 | `plotting/slow_manifold.py`（面板 B：settled 青色 + moving 橙色） |

---

## 附：当前状态与启用路径

- 代码已重做并通过测试（`model-dev/learning`，196 passed）；两组诊断在实验中
  仍默认关闭（`timescale_separation: false` 等）。
- 等 N=120 von-Mises 与 heterogeneous 训练完成后，先在 best weights 上跑
  `test_vafidis_toy --run-dir ...`（常规诊断组），再按 §8 对比两网络的
  `slow_manifold_eta_theta_deg_s` / `spectral_gap_min` /
  `slow_mode_tangent_alignment_median` / `basin_entropy` / `settled_fraction_median`。
- 确认后才在 diagnostics YAML 打开 `pva_spectrum_and_visualization`、
  `slow_manifold`、`timescale_separation`。
