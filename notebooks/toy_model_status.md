# Toy Model 研究复核与一周组会计划（2026-08-24）

> 复核范围：N=60、seed 42/43/44 的三次 80 ks 训练及 2026-08-24 的重新诊断；
> Vafidis 原论文与作者源码；Sagodi、Clark、Noorman 等相关工作。
> 本文替代 2026-08-23 的阶段稿。旧稿保存在
> `toy_model_status_2026-08-23_pre_audit.md`，其中的 OU 单位、HR 偏移和拓扑措辞不可继续引用。

## 0. 先给结论

这批结果已经足够做一次有内容的组会，但最可信的故事不是“已经学出了连续吸引子”，而是：

> **Vafidis 型局部预测学习在三个 seed 中都能自组织出高度相似的环形权重结构，
> 并在一段训练窗口内获得中高速的短时路径积分能力；然而局部训练误差与冻结后的行为性能逐渐脱钩，
> N=60 网络保留明显的低速 pinning，60 s 累积误差很大。最新诊断在修复索引错误后均运行完成，
> 但三组低速状态都只覆盖 22.5–25% 的角度网格、集中成 27–30 个 preferred-heading-aligned clusters，
> 且 operational timescale criterion 全部失败。因此当前证据支持“约 30 个势阱的有限 N quasi-CAN”，
> 不支持一条已建立的 persistent slow ring；更精确的全状态几何仍待校准后的诊断。**

可以分三级陈述：

| 结论 | 当前证据强度 | 安全措辞 |
| --- | --- | --- |
| 局部规则可重复地产生 Vafidis 式权重 motif | 强，三 seed 一致 | “机制层面的部分复现” |
| best snapshot 有短时 PI 能力，但低速和长时性能差 | 强 | “learned transient competence with pinning” |
| 训练后期局部目标与行为目标脱钩 | 强，三 seed 一致 | “功能最优点早于局部误差最优点” |
| N=60 的渐近动力学 | endpoint、低速 clusters 和 timescale test 均指向强 phase locking；几何证明仍不完整 | “约 30 势阱的离散 quasi-CAN；未建立 persistent slow ring” |
| 已严格复现 Vafidis 论文或作者 release | 不成立 | “paper-equation-inspired numerical variant” |

本周最重要的选择是：**先修诊断和补齐 matched、held-out 的冻结权重评估，不再启动新的 80 ks 训练。**

## 1. 数据身份与审计边界

三次原始训练位于：

- `model-release/learning/runs/vafidis_baseline/20260822-234631_vafidis_toy_42`
- `model-release/learning/runs/vafidis_baseline/20260822-234739_vafidis_toy_43`
- `model-release/learning/runs/vafidis_baseline/20260822-234826_vafidis_toy_44`

根目录 `runs/vafidis_baseline/` 下的同名目录是迁移副本，并在 2026-08-24 用新诊断代码重新后处理；
它们不是三次新的训练。20:09–20:21 完成的最新批次中，三组均为 14 diagnostics completed / 0 failed，
11 figure groups completed / 0 failed。原始 pro 与根目录副本共有的训练、权重、gain、constant PI、OU、
bump 和旧 endpoint 数值逐值相同；新批次主要增加 83 个 slow/Ramesan/timescale 指标。
下文的训练和基本行为指标可由两处相互校验，新 slow/timescale 结论以根目录最新副本为准。

三次训练除 `simulation.seed` 外配置相同：

- 60 个 HD、60 个 HR 单元；HD 两两共享方向，实际只有 30 个独特 preferred headings，间隔 12°；
- von Mises 视觉教师：amplitude 4、baseline 5、kappa 11.111；
- 训练 OU：stationary std 225°/s、tau 0.5 s、无 clip；
- 80,000 s，神经步长 0.125 ms，proximal `exact_linear`；
- `block_multirate`，权重每 10 ms 更新一次，即每 80 个神经步更新一次；
- eta 50、tau_delta 0.1 s；无 weight clipping、decay、balance、symmetry 或 zero-diagonal 约束；
- 训练期 early stopping 和 checkpoint selection 均关闭。

训练结束后，runner 又离线比较每 1.6 ks 的 snapshot，并用 4 个 heading、7 个速度
（0、±15、±30、±75°/s）、每个 5 s 的同一组 probe 选择 `best`。
此后默认诊断用 `best_weights.npz`，并把它另存为容易误读的 `trained_weights.npz`；
真正的 80 ks 权重是 `final_weights.npz` / `training_selected_weights.npz`。
因此“best”含有 post-hoc selection optimism，尚未在独立 headings、velocities 和 OU seeds 上验证。

### 1.1 尚未闭合的 provenance

run 内保存了 resolved config，但没有训练时的 Git commit、dirty diff、import path、环境锁定文件或源码 hash；
现有 release manifest 又是在训练和原始诊断完成约 8 小时后生成，且记录 `git_dirty=true`。
因此可以接受操作者关于 pro 环境的记录，但无法从 artifact 独立证明当时加载的精确源码。
后续 run 至少应保存：commit、dirty patch/hash、Python executable、`learning.__file__`、config hash、
weight source/time 和诊断代码 hash。

## 2. 三 seed 的定量结果

### 2.1 训练窗口：行为最优点明显早于 80 ks

| seed | best 时刻 (ks) | best 5 s 平均绝对 PI 误差 | final | best / final stall fraction | best / final RMS 速度偏差 (°/s) | best / final HR:HD norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 35.2 | 9.27° | 31.25° | 20.8% / 50.0% | 5.51 / 14.06 | 2.397 / 2.398 |
| 43 | 20.8 | 8.07° | 39.46° | 12.5% / 62.5% | 4.48 / 17.83 | 2.245 / 2.387 |
| 44 | 16.0 | 11.23° | 35.00° | 4.2% / 58.3% | 4.93 / 16.39 | 2.196 / 2.384 |

这不是简单的“局部规则没有收敛”。记录的局部 absolute learning error 从行为 best 到 final 仍继续下降：

| seed | 行为 best 时的 local error (spikes/s) | 80 ks final | 变化 |
| ---: | ---: | ---: | ---: |
| 42 | 0.957 | 0.638 | -33% |
| 43 | 1.077 | 0.585 | -46% |
| 44 | 1.402 | 0.739 | -47% |

这项 10 s window absolute learning error 分别到 76.8、79.2、77.6 ks 才达到最小，
远晚于 16–35.2 ks 的行为最优点。
权重范数仍在增长，但训练尾段的归一化增长率约为 3–5×10^-6/s，与库内作者成熟网络同量级，
没有证据把问题归因于数值爆炸。更准确的解释是：

1. 局部 prediction-error 目标并不等价于 darkness PI、低速可动性或长时稳定性；
2. 当前 post-hoc checkpoint 只是在有限 probe 上寻找功能窗口；
3. “80 ks 过训练”可作为当前实现的功能描述，不能推广成对论文规则的一般结论。

### 2.2 best snapshot：整体 gain 好看，但低速和长时结果差

| seed | ±500°/s gain / R² | 60 s 常速度 sweep 的终点 RMS 误差¹ | 单条 60 s OU 的 RMS PI 误差 | 24-trial OU 末时刻 SD² | 系统漂移 (°/s) | D (deg²/s) | bump final PVA³ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.938 / 0.9993 | 479° | 58.4° | 131.9° | 0.79 | 289.8 | 0.955 |
| 43 | 0.936 / 0.9986 | 486° | 62.6° | 120.8° | 0.40 | 243.4 | 0.951 |
| 44 | 0.935 / 0.9986 | 396° | 159.6° | 127.1° | 3.23 | 269.2 | 0.948 |

1. 常速度 sweep 指跨测试速度的 **unwrapped** 60 s 累积误差 RMS，不应与 wrapped circular endpoint error 混写。
2. `ou_pi_ensemble_final_pi_error_std` 在文件中以 **rad** 保存。旧稿把 2.1–2.3 rad 误写成 2.1–2.3°；
   正确值是约 121–132°。这是本次最重要的数值勘误。
3. bump maintenance 只从 `theta=0` 的单个初相位测试 5 s，只能证明该相位附近存在稳定 bump，不能证明整环连续。

整体线性回归主要由大速度点决定。三个 seed 的 depinning threshold 都约为 30°/s；
在 ±15°/s probe 上，不同 heading 有 0–75% 的 trial 被 pin 住。
所以“gain≈0.94、R²≈0.999”只能说明中高速映射近线性，不能代替低速响应图，更不能推出 60 s PI 准确。

### 2.3 学到的权重结构是最稳健的正结果

- HD recurrent local symmetry score 为 0.976、0.956、0.957；三 seed 的完整 HD 权重相关约 0.990–0.997；
- HR→HD 权重跨 seed 相关约 0.998–0.999；
- 左、右 HR 的兴奋性 source offset 分别约为 -72° 和 +72°，三 seed 很一致；
- 若用包含抑制项的 abs-weight COM，偏移只有约 -18° / +18°。

因此旧稿的“HR→HD +48° 偏移”没有当前 metric 支持，应删除。
更可靠的图应同时画正权重峰、负权重谷和 circular profile，并清楚写出偏移的定义。

## 3. 与 Vafidis 论文和作者源码的关系

当前模型在网络构造、膜/树突方程、局部学习规则和多数物理参数上相当忠实；
关键差别集中在数值协议、源码 bug 是否保留、权重选择和测试统计量。

| 项目 | Vafidis 论文 | 作者源码 | 当前三次 run | 影响 |
| --- | --- | --- | --- | --- |
| N 与角度 | 60 HD + 60 HR；30 个独特方向 | 相同 | 相同 | 可比 |
| 训练长度 | 80 ks，12 runs | 80 ks | 80 ks，3 seeds | 单 run 可比，统计强度不同 |
| 神经积分 | 0.5 ms forward Euler | 0.5 ms ordered Euler | 0.125 ms `exact_linear` | 不是逐实现复现 |
| plasticity 时钟 | 每个 0.5 ms step 在线更新 | 同左 | 每 10 ms block 更新 | 长时科学等价尚未验证 |
| HD→HR low-pass | tau=65 ms | 更新式漏乘 dt，等效约 32.5 ms | 正确实现 65 ms | 忠实论文，但不忠实 literal release |
| 初始权重尺度 | 文字写 1/sqrt(60) | 实际 1/sqrt(120) | 1/sqrt(120) | 忠实源码、偏离论文文字 |
| 诊断权重 | 成熟的 80 ks final | 最后一帧 | post-hoc `best` | 当前结果不是 80 ks final 表现 |
| gain 协议 | 约 5 s/速度，强调 ±500 和 <30°/s | 5 s | cue 20 s + darkness 60 s | 形状可参考，数值不可逐点等同 |
| OU PI | 1000×60 s，限制 ±500°/s | 直接 dark，wrapped endpoints | 24×60 s，无 clip，release-relative unwrapped trace | D 与误差分布不可直接比较 |

项目自己的 `2026-08-19_multirate_training_integrator.md` 已明确：现有短时 matched-stream 测试只是
implementation-scale 验证，不能证明 10 ms block-multirate 与原始 online learning 在 80 ks 上科学等价。
所以组会中应把本模型称作 **“Vafidis paper equations 的当前数值变体”**，而不是“原论文复现”。

### 3.1 作者 released network 是现成正对照

库内还保留了作者的 12 个 80 ks 网络和现成冻结诊断，可作为 ground-truth control。
其中标准 final 网络在作者协议下得到 1000×60 s OU endpoint RMS 38.73°、D≈24.5 deg²/s、
60 s bump drift 0.53°；把其 gain 数据限制到 |v|≤500°/s 后，dark slope=1.0065、R²=0.9988。
它同样在约 |v|<30°/s 出现 flat/pinning 区。作者协议与当前协议仍不同，不能直接做显著性比较，
但这组 control 说明“论文现象能由作者 released network 在作者统计口径下恢复”，也提供了本周 matched overlay 的基线。
数据入口为 `Vafidis/diagnostics/output/`，12 个权重位于
`Vafidis/savefiles/trained_networks/Parallel/Main_Net/`。

### 3.2 论文并不要求 N=60 是光滑中性环

Vafidis 把 N=60 的成熟网络称为 quasi-continuous attractor，同时明确承认有限 N 下存在彼此分开的
discrete attractors/basins，并报告 |v|<30°/s 的 flat/pinning 区；增大到 N=120/240 后该区域减小。
因此约 25–30 个稳定相位可能正是 30 个独特 preferred headings 的有限 N 粗粒化，不能仅凭 fixed-point 数称为失败。

另一方面，这也不等于已经证明了现代意义上的连续吸引子。Sagodi 的慢环可以同时包含交替的 stable/saddle
fixed points；“有离散 FP”和“存在闭合、法向吸引的慢环”并不互斥。今后应把问题拆成三条轴：

1. **几何轴**：完整冻结 Markov state 中是否有闭合、吸引、法向快收缩的慢环；
2. **切向轴**：沿环是零流、慢 stable/saddle pairs，还是 limit cycle；
3. **功能轴**：5–60 s 的误差、低速 pinning、phase-dependent velocity response 和扰动恢复。

Noorman 的零输入相位锁定、小速度不能更新、响应随 bump phase 改变，是很敏感的 finite-N discreteness 读出，
但仍不是状态空间几何证明。Clark 则说明 disorder 不必破坏连续流形，但其 1D HD 示例在 N≈1000 已接近目标环；
旧稿中的“Clark 需要 N>10^4”是错误概括，N=25,000 对应的是二维 grid/toroid 示例。

## 4. 当前 attractor / slow-manifold 结果能与不能说明什么

### 4.1 endpoint map classifier 只能作为探索性有限时结果

原始 pro 后处理与最新完整重跑逐值一致；5 s PVA endpoint classifier 给出：

| seed | PVA attracting / repelling / unresolved | peak decoder attracting / repelling / unresolved | 安全解释 |
| ---: | ---: | ---: | --- |
| 42 | 25 / 25 / 0 | 6 / 3 / 3 | PVA 下呈规则离散 phase locking，但 decoder 依赖大 |
| 43 | 2 / 24 / 22 | 7 / 6 / 1 | 大量 transition 未分类，不能称“近连续” |
| 44 | 5 / 23 / 18 | 2 / 2 / 0 | 同上 |

PVA 与 peak/overlap decoder 给出的 root 数差异很大；seed 43/44 的 stable/unstable 又不交替。
此外初始化用了 amplitude 24 的强 cue，而训练 cue 只有 4，所有 trial 都出现多个接近饱和的 HD bins。
所以这些结果的正确标题是 **finite-time endpoint-map classifications**，不能据此写“同参数学出不同拓扑”。

### 4.2 修复后完整三-seed 重诊断（截至 2026-08-24 20:22）

当天较早的一批后处理曾因 candidate absolute/relative index 混用而得到 12 completed / 2 failed，
并暴露出 stale-cache 风险。当前 `model-release` 已把时间分槽改为保存 candidate position、末尾只映射一次，
并有对应回归测试；最新三组均为 14 completed / 0 failed，相关 NPZ 和图也都在本批次重新生成。
因此早先的 IndexError 和“空 NPZ/旧图”只能作为工程审计历史，不能再描述当前产物。

但是 **operational completion 不等于 scientific criterion passed**：

| seed | occupied bins / 120 | angular support | low-speed clusters | Ramesan stable / unstable phase | normal p90 / tangential p10 | timescale ratio | slow-ring / timescale criterion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 28 / 120 | 0.233 | 28 | 30 / 28 | 0.204 s / 0.050 s | 0.246 | fail / fail |
| 43 | 30 / 120 | 0.250 | 30 | 30 / 30 | 0.215 s / 0.050 s | 0.233 | fail / fail |
| 44 | 27 / 120 | 0.225 | 27 | 30 / 34 | 0.218 s / 0.050 s | 0.229 | fail / fail |

- 三组 `slow_manifold_fit_succeeded=0`、`fit_failure_is_insufficient_coverage=1`；
  拟合后的 `manifold_state/angular_flow/Jacobian spectrum` 数组为空，因而不存在可报告的 fitted slow ring；
- 27–30 个低速簇到最近的 30 个 preferred headings 的距离都约为 1.5°，即 3° histogram bin 的中心偏移；
- 直接对 360 条 5 s PVA endpoints 做 gap-based cluster check，在合理 2–8° 阈值下也得到约 27–30 个 plateau，
  与图上的约 12° 台阶一致；
- Ramesan dense phase summary 在三 seed 都解析出约 30 个 stable phase；slow bins 内的 settled-fraction median
  为 0.856、0.857、0.873，而全相位 median 为 0，符合“少数相位驻留、其余相位快速穿越”。
  但 seed42/44 的 stable/unstable 不完全配对，root 数又随平滑变化，不能把 58/60/64 当精确拓扑计数；
- 30 个 candidate Jacobian anchors 的 leading real eigenvalue 均明显为负（约 -7.21…-3.35 s^-1），
  支持抽中的点是强稳定 fixed points；但 anchors 只覆盖 23/28、23/30、25/27 个簇，且没有完整谱隙，
  不能替代每簇 root solve 或 tangent/normal spectrum；
- timescale ratio 的预设通过阈值是 10，三组却只有约 0.23–0.25，当前 operational criterion 明确未通过。
  其中 0.05 s tangential p10 受 0.1 s 采样、3° 阈值和量化 decoder 限制，normal distance 又相对 cue-defined
  closed curve 而非已识别 autonomous manifold；所以它是强负面 QC，不是独立的 attractor 拓扑定理。

综合 endpoint staircase、preferred-heading-aligned low-speed clusters、低速 depinning 和 timescale failure，
最节约假设的解释是：**三 seed 都学到了约 30 势阱的 pinned/discrete quasi-CAN；25/2/5 的 PVA root-counter
差异主要来自 transition classifier，而不是三种已被证明的拓扑。**

### 4.3 仍未解决的诊断边界

1. `trajectory_autonomous_speed = norm(full_state_flow)` 对包含 firing rate、电流和电压的 300 维向量直接取范数，
   但 schema/YAML 仍把 `slow_manifold_speed_floor=5e-4` 称为 rad/s。三 seed 的 threshold 都被这个 mixed-unit
   floor 截断，因此 candidate speed 不能解释成物理角速度；应改成无量纲归一化 norm，或直接使用 decoded phase speed。
2. 时间分槽已不越界，但 seed42 的 4096 个 candidate rows 中仍有 6 个重复，需补唯一性保证。
3. failure recorder 仍不会删除旧 NPZ，绘图也不会主动清旧图。本批次所有相关 mtime 都是新的，所以当前结果不是旧 cache；
   但未来仍应写入 per-diagnostic provenance/hash，并只从本次成功产物目录绘图。
4. `model-release-anneal` 仍保留旧的 candidate-index bug，不能用 `pro-anneal` 跑这组诊断。
5. coverage 不足会漏掉 measure-zero saddle/repelling states；因此当前结果强烈反对一个“处处慢”的环，
   但不是关于所有可能 invariant connections 的拓扑定理。完整结论仍需 boundary continuation、全状态 root solve
   和在已验证 anchors 上的 Jacobian tangent/normal spectrum。

## 5. 当前最合理的科学解释

### 5.1 已被三 seed 支持

1. 局部 predictive rule 能从随机权重中稳定形成 Vafidis 预期的 HD 局部环形结构和左右镜像 HR shift motif。
2. 学到的网络能在一个 checkpoint 窗口内把中高速 angular velocity 转成近线性的 bump motion。
3. N=60 的低速 pinning 很强；三 seed 都出现约 27–30 个 preferred-heading-aligned 低速簇，
   和论文对 finite-N discrete basins 的描述定性一致。
4. 局部 prediction error 与最终关心的 darkness behavior 明显不一致；用 local error 选模型并不安全。

### 5.2 当前不支持

1. 不能说当前 toy model 已实现“accurate long-horizon PI”：60 s constant/OU 误差过大且 seed 44 漂移明显。
2. 不能用 wide-range gain 和 R² 证明低速积分或连续吸引子。
3. 不能把 best snapshot 当作训练自然收敛到的 80 ks final。
4. 不能把当前 D=243–290 deg²/s 与论文 24.5 deg²/s 作直接成败比较：trial 数、速度 clip、初态、
   wrapped endpoint 与 release-relative unwrapped trace 都不同。
5. 不能把 endpoint classifier 的 unresolved 区间当作 near-continuous 的正证据；当前跨方法一致的结果反而是
   三 seed 都有约 30 个 plateau/低速相位，而 fitted persistent slow ring 与 timescale criterion 均未通过。

### 5.3 最值得检验的三个原因

按优先级排列：

1. **评估与模型选择错配**：best 在同一小组 probe 上被选择，尚未做 held-out 验证；local error 又与 PI 脱钩。
2. **训练数值协议错配**：10 ms block 更新与 0.5 ms online rule 的长时等价没有建立。
3. **有限 N roughness**：30 个独特方向天然容易形成 phase locking，整体 gain 会掩盖低速和相位依赖误差。

视觉异质性、visual annealing、N=120、Clark projection 和 meta-learned rule 都是合理后续方向，
但在这三个基础问题闭合前继续扩展，会把“训练问题、评估问题还是吸引子问题”混在一起。

## 6. 一周内的执行计划

目标不是在一周内解决全部理论问题，而是得到 **4 张机制/行为主图 + 1 张有明确方法边界的状态空间负结果图**，
并形成一条不会被协议或单位问题轻易推翻的叙事。

| 时间 | 工作 | 交付物 | 继续/止损条件 |
| --- | --- | --- | --- |
| Day 1 | 冻结 20:09–20:22 最新产物和 manifest hash；修剩余的 mixed-unit floor、candidate uniqueness 与失败缓存语义 | 版本化诊断目录、code/config hash、测试报告 | 新速度量必须有一致单位；失败项不得留下可被绘图读取的旧文件 |
| Day 2 | 不重训；对三 seed 的 best/final 和作者网络跑完全相同的轻量 frozen eval | held-out heading/velocity 表；作者网络 reference control | best 若只在 selection probes 好，就把“checkpoint 最优”降为选择过拟合 |
| Day 3 | 做严格的 paper-style 行为对照 | 5 s gain；±100°/s 放大；clipped ±500°/s OU | OU 先用 100 trials 验证，再流式跑 1000，避免保存数 GB 全状态 |
| Day 4 | 做 endpoint duration/decoder/smoothing sensitivity；用修正后的 phase-speed selector 复核低速 coverage | cluster/plateau 稳健性；可选 boundary root solve | unresolved>10%、精确 root 数对参数变化>20% 或 anchors 未覆盖各簇，就只报约 30 phase-locking，不报精确 FP 数 |
| Day 5 | 统一作图和统计口径 | 5 张主图；所有角度转成 degree；CI/SD 和 wrapped/unwrapped 写清 | 不再加新模型；先让同一数字在 JSON、图注、正文三处一致 |
| Day 6 | 做 8–10 页组会 slide，删掉不能 defend 的 panel | 完整讲稿和 appendix | 保留 phase-locking 负结果；删除没有完整谱隙支撑的 slow-mode/topology claim |
| Day 7 | 复核、彩排和缓冲 | 15–20 分钟版本；问题清单 | 只修错，不启动新长训练 |

### 6.1 冻结评估的最小 protocol

三 seed、作者网络、best/final 都使用同一套输入流：

1. **checkpoint held-out**：heading 使用与选择集错开的网格，velocity 至少密集覆盖 -75…75°/s，
   另留独立 OU seeds；选择集与报告集严格分离；
2. **gain**：同时画全景 ±500°/s 和低速 ±100°/s inset，报告 phase-dependent residual，不能只报 slope/R²；
3. **OU**：并列保留两种口径——复刻论文的 clipped、wrapped endpoint distribution，以及当前更严格的
   release-relative unwrapped trace；分别报告 systematic drift、variance/D 和置信区间；
4. **endpoint**：至少 360 初相位，并把 30 个独特 preferred headings 叠在图上；改变 autonomous duration、
   bin 和 smoothing，检查 root 是否稳定；
5. **slow ring**：使用完整冻结 Markov state、全 dynamics Jacobian、tangent/normal alignment、angular coverage
   和独立长轨迹 root 验证；不要用单一 decoder endpoint 数代替。

## 7. 建议的组会图版

### 图 1：局部规则学出了什么？

- 一张自己绘制的模型/学习规则简图；
- seed 42 的 HD→HD、LHR→HD、RHR→HD 矩阵；
- 三 seed circular profile 的 mean±range，并标 ±72° excitatory offsets；
- untrained 和作者 released network 只作形态 reference，不写逐数值复现。

**一句话结论**：局部预测学习跨 seed 稳定地产生了预期的环形和速度偏移结构。

### 图 2：训练目标与行为目标何时脱钩？

同一横轴画三 seed 的：local prediction error、5 s frozen PI error、stall/depinning、HD/HR norms；
标出各自 best 和 80 ks final，再用 paired bars 总结 best→final。

**一句话结论**：局部误差继续变好时，darkness 行为已经恶化；model selection 是研究问题的一部分。

### 图 3：全范围 gain 如何掩盖低速 pinning？

- 主图：-75…75°/s 的 decoded-vs-commanded，保留每个 heading 的点；
- inset：±500°/s 全景和斜率；
- 作者网络、current best、current final 使用完全相同 protocol；
- 可加 phase-dependent velocity residual heatmap。

**一句话结论**：中高速传递近线性，但 N=60 在零速附近存在明显势垒。

### 图 4：短时可用是否能转化成长时 PI？

- 三 seed 60 s constant PI error trace，明确标 unwrapped degree；
- clipped OU 的代表轨迹和 ensemble median/50%/90% 区间，避免只画 SEM；
- 10/20/…/60 s endpoint distribution；systematic drift 与 diffusion 分开；
- paper-style metric 和 current metric 分面，不把 D 混在同一口径。

**一句话结论**：短时速度映射并未自动带来稳定的长时积分，且 seed 间 systematic drift 不同。

### 图 5：约 30 重 phase locking 与未通过的 slow-ring criterion

- 三 seed endpoint maps 和 slow-candidate occupancy，标题明确写 finite-time / trajectory-conditioned；
- 叠加 30 个 preferred headings，报告 28/30/27 个低速簇与 22.5–25% coverage；
- 用一个小表报告 timescale ratio 0.23–0.25 vs criterion 10；
- 不展示精确 PVA root 数或 Ramesan “effective potential” 作为主证据；没有完整谱隙就不画 slow-mode 结论。

**一句话结论**：三个 seed 都更像约 30 势阱的 finite-N quasi-CAN，当前 operational tests 不支持 persistent slow ring。

已有图中，权重矩阵和 `training_snapshot_frozen_pi_error.png` 可作为作图素材；
`velocity_gain_curve.png` 只能做全景 inset；OU 图需要把 rad 转 degree 并用分位区间重画。
根 `runs/` 最新的 trajectory、slow-candidate 和 timescale 图确为本批次新产物，可作为重画素材；
endpoint 图目前有标题重叠，timescale 的 0.05 s 又是 resolution-limited，均不宜原图直接上组会。
Ramesan 1-D potential 只放 appendix，并明确它不是物理能量或拓扑证明。

## 8. 建议的 10 页叙事

1. 问题：局部学习能否形成一个可用、近连续的 HD 积分器？
2. Vafidis 模型与当前数值实现；把与论文/source 的差异提前说清。
3. 三 seed 和 checkpoint 审计；解释 best 与 final。
4. 图 1：跨 seed 的权重 motif。
5. 图 2：local objective 与 behavior 脱钩。
6. 图 3：中高速 gain 与低速 pinning。
7. 图 4：60 s constant/OU 累积误差。
8. Vafidis、Sagodi 语境下如何区分有限 N quasi-CAN 与完整慢环。
9. 图 5 与诊断 integrity：为什么当前可称约 30 势阱的 discrete quasi-CAN，却不能称 persistent slow ring。
10. 一周后/组会后的判别实验与决策树。

组会结尾不要说“模型失败了”或“已经得到连续吸引子”。更准确的总结是：

> **机制形成是可重复的，短时功能存在但评价依赖 checkpoint；三个 seed 都呈约 30 重 phase locking，
> 低速和长时仍不够好。下一阶段首先要把训练目标、数值协议与全状态动力学三者分开。**

## 9. 组会后的决策树

1. 若 current best 在 held-out probes 明显退化：先重做 checkpoint metric，不启动 N=120。
2. 若作者网络在相同 current protocol 下好、current weights 差：优先查训练时钟、Eq9 和初始化/尺度差异。
3. 若 `single_clock` 与 10 ms `block_multirate` 的 matched-stream/full-training 结果明显不同：
   block approximation 是首要机制变量；至少做三 seed matched 80 ks 后再谈学习规则本身。
4. 若 matched dynamics 后 N=60 仍稳定 pinning，而作者网络也类似：这更像有限 N 限制；此时再做 N=120/240。
5. 若 slow-ring coverage、normal spectral gap 或扰动恢复不过关：称为 pinned/discrete quasi-CAN，
   不使用 continuous/ghost 标签。
6. 只有基础 matched baseline 闭合后，才比较 heterogeneous cue、visual anneal、Clark-style restoring dynamics
   或 Bell 式 meta-learned local rule。

本次没有重新审计旧 N=120 heterogeneous、正在/已运行的 von-Mises N=120 或 visual-anneal 结果；
它们不应混入这次 N=60 复核的主结论。审计前的相关记录仍可在旧稿查阅，但需要按本文的单位、protocol
和拓扑标准重新验证。

## 10. 本地证据入口

- Vafidis 论文：`references/Learning accurate path integration in ring attractor models of the head direction system.pdf`
- 作者源码：`Vafidis/original/fly_rec.py`、`Vafidis/original/generate_plots.py`
- 当前模型与学习规则：`model-release/learning/src/learning/models/vafidis_toy.py`、
  `model-release/learning/src/learning/plasticity/predictive_local.py`
- 当前训练/诊断 runner：`model-release/learning/src/learning/experiments/run_vafidis_toy.py`
- 慢流形分析：`model-release/learning/src/learning/analysis/slow_manifold.py`
- 诊断原理笔记：`notebooks/slow_manifold_diagnostics.md`
- Sagodi：`references/sagodi2024_back_to_continuous_attractor/`
- Clark：`references/clark2025_symmetries/`
- Noorman：`references/Maintaining and updating accurate internal representations of continuous variables with a handful of.pdf`

---

**当前推荐下一动作**：先冻结最新三-seed 诊断与代码 hash，用半天补齐 mixed-unit、candidate uniqueness
和 stale-cache QC；同时立即启动现有 best/final 与作者网络的 matched、held-out 冻结评估。
本周不再启动新 80 ks 训练。这样最快能把“学到了什么、哪里失效、下一步怎样判别”变成五张可讲的图。
