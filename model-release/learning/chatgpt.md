我赞成引入 scale factor，但需要先把不同来源的量分开：**不能给整个模型统一乘一个 (1/N) 或 (1/\sqrt N)**。最合理的做法是区分：

[
\boxed{\text{coherent / structured population sum}\sim \frac1N}
]

[
\boxed{\text{zero-mean random matrix contribution}\sim \frac1{\sqrt N}}
]

[
\boxed{\text{direct current received by each neuron}\sim O(1)}
]

其中，recurrent excitation、HR(\to)HD 输入和 population inhibition 属于第一类；零均值随机 connectivity disorder 属于第二类；visual cue 和逐神经元 OU current noise 通常属于第三类。

严格来说，Clark 的原始动力学方程并没有把 (1/N) 显式写在求和号前，而是将它嵌入低秩连接

[
J=\frac1N UV^\top
]

以及均匀抑制项 (-c/N) 中。因此 (\sum_jJ_{ij}\phi_j) 在 (N\to\infty) 时仍然是 (O(1))。这正是 softplus 这类非饱和激活函数能够使用的前提之一；数据网络中还通过 (J_{ij}\rightarrow J_{ij}-c/N,\ b=c) 稳定 population mean mode。  

## 1. 把连接矩阵改写为“intensive kernel”

建议不要让代码中的权重矩阵同时承担“连接形状”和“有限规模归一化”两个角色。定义 (O(1)) 的 intensive kernels：

[
K^{\mathrm{rec}}\in\mathbb R^{N_{\mathrm{HD}}\times N_{\mathrm{HD}}},
\qquad
K^{\mathrm{HR}}\in\mathbb R^{N_{\mathrm{HD}}\times N_{\mathrm{HR}}},
]

然后把 distal current 改写为

[
\begin{aligned}
\tau_s\dot I_i^{d}
=&-I_i^{d}
+
\frac{g_{\mathrm{rec}}}{N_{\mathrm{HD}}}
\sum_{j=1}^{N_{\mathrm{HD}}}
K^{\mathrm{rec}}*{ij}r^{\mathrm{HD}}*j\
&+
\frac{g*{\mathrm{HR}}}{N*{\mathrm{HR}}}
\sum_{k=1}^{N_{\mathrm{HR}}}
K^{\mathrm{HR}}_{ik}r^{\mathrm{HR}}_k
+
I^{\mathrm{mean}}_i
+
\sigma_d\xi_i^d .
\end{aligned}
]

这里的 (K) 描述 continuum connectivity kernel，(g_{\mathrm{rec}}) 和 (g_{\mathrm{HR}}) 描述两个通路的总强度。增加 neuron count 相当于对同一个积分进行更精细的离散化：

[
\int_0^{2\pi}\frac{d\theta'}{2\pi}
K(\theta,\theta')r(\theta')
\quad\longrightarrow\quad
\frac1N\sum_jK_{ij}r_j.
]

Vafidis 原模型因为固定使用 (N_{\mathrm{HD}}=N_{\mathrm{HR}}=60)，直接写成了 (W^{\mathrm{rec}}r^{\mathrm{HD}}+W^{\mathrm{HR}}r^{\mathrm{HR}})，没有解决跨 (N) 的标度问题。 

对于 HD(\to)HR 通路要区别处理。Vafidis 中它基本是 one-to-one mapping，因此每个 HR cell 只接收特定 HD cell 的输入，这种连接不需要 (1/N)。只有将它推广成 dense population projection 时，才应加入 (1/N_{\mathrm{HD}})。

为了兼容目前以某个参考规模调好的参数，也可以暂时采用

[
s_N=\frac{N_{\mathrm{ref}}}{N},
]

例如原基准为 (N_{\mathrm{ref}}=32)，则

[
I_{\mathrm{rec}}(N)
===================

\frac{32}{N}W^{\mathrm{rec}}r.
]

但从代码清晰性而言，显式写成 `K @ r / N` 更好。

## 2. 用 population-mean inhibition 稳定 softplus

对于非饱和 softplus，不建议继续使用与 (N) 无关的 summed inhibition。可以把 Clark 的 mean-mode stabilization 改写成更直观的形式：

[
I_i^{\mathrm{mean}}
===================

c\left(r_0-\bar r_{\mathrm{HD}}\right),
\qquad
\bar r_{\mathrm{HD}}
====================

\frac1{N_{\mathrm{HD}}}
\sum_jr_j^{\mathrm{HD}}.
]

这里 (r_0) 是目标 population mean firing rate。若 (r_0=1)，它就与 Clark 的

[
J_{ij}\rightarrow J_{ij}-\frac cN,\qquad b=c
]

等价，因为额外输入为

[
c-\frac cN\sum_jr_j=c(1-\bar r).
]

这样做有三个好处：不同 (N) 下工作点基本不变；softplus 不会因为 recurrent sum 随 (N) 增长而进入越来越高的 firing-rate 区域；Vafidis prediction error 中 proximal 与 distal firing-rate prediction 的量级更容易保持一致。

不建议在每个时间步直接对神经活动做归一化，因为那会改变原始 two-compartment prediction-error dynamics。应通过输入标度和 mean-mode feedback 控制工作点。

## 3. GP visual cue 本身不要乘 (1/N)

Clark 式 GP cue 是每个 HD neuron 直接接收的外部电流，因此其 per-neuron amplitude 应保持 (O(1))，不应因为神经元数量增加而变小。否则大网络中的 teacher 会系统性变弱，学习条件也随 (N) 改变。

先按 Clark 的方式对每条 tuning curve 做 angular mean normalization：

[
q_i(\theta)
===========

\frac{
\operatorname{softplus}!\left[\beta(x_i^\star(\theta)-b)\right]
}{
\left\langle
\operatorname{softplus}!\left[\beta(x_i^\star(\theta)-b)\right]
\right\rangle_\theta
}.
]

Clark 的 (Z[x^\star]) 正是为了让每个生成的 tuning curve 在 (\theta) 上具有 unit mean。

但只做这一层还不够：有限 (N) 时，不同方向下的 population mean 与 population modulation RMS 仍会波动。建议为跨 (N) benchmark 增加一个 matched-drive normalization：

[
\mu_N(\theta)=\frac1N\sum_iq_i(\theta),
]

[
\delta q_i(\theta)=q_i(\theta)-\mu_N(\theta),
]

[
s_q^2=
\left\langle
\delta q_i(\theta)^2
\right\rangle_{i,\theta},
]

[
I^{\mathrm{vis}}_i(t)
=====================

I_{\mathrm{vis},0}
+
A_{\mathrm{vis}}
\frac{\delta q_i(\theta(t))}{s_q}.
]

于是：

* (I_{\mathrm{vis},0}) 控制 proximal baseline；
* (A_{\mathrm{vis}}) 控制异质视觉调制的 RMS；
* 两者均与 (N) 无关；
* 每个方向上的 accidental common-mode drive 被移除。

这应该作为“严格控制输入强度”的主实验。与此同时保留一个 raw-GP condition：

[
I_i^{\mathrm{vis}}
==================

I_{\mathrm{vis},0}
+
A_{\mathrm{vis}}
\frac{q_i(\theta)-1}{s_q},
]

它保留有限样本造成的 population imbalance。raw condition 中这类 imbalance 本身自然按 (N^{-1/2}) 衰减，它可能正是 pinning、dead zone 或噪声改善性能的来源。不要再额外乘一个 (1/\sqrt N)，否则会把自然有限规模效应压成 (1/N)。

还应固定：

[
\sigma_{\mathrm{GP}},\ \sigma_{\mathrm{vis}}
]

的单位为 radian，而不是“多少个 neuron bins”。否则改变 (N) 时，cue 的物理角宽也在改变。

## 4. 学习规则也必须采用一致参数化

Vafidis 的局部规则为

[
\Delta W_{\mathrm{pre,post}}
============================

\eta,
\left[
f(V^a_{\mathrm{post}})
----------------------

f(pV^d_{\mathrm{post}})
\right]
P_{\mathrm{pre}}.
]



若代码存储的是 intensive kernel (K)，推荐直接写成

[
\dot K^{\mathrm{rec}}_{ij}
==========================

\eta_{\mathrm{rec}}e_iP_j^{\mathrm{HD}},
]

[
\dot K^{\mathrm{HR}}_{ik}
=========================

\eta_{\mathrm{HR}}e_iP_k^{\mathrm{HR}},
]

[
e_i=f(V_i^a)-f(pV_i^d),
]

而在计算电流时才使用 (K/N)。这样：

* (\eta_{\mathrm{rec}}) 和 (\eta_{\mathrm{HR}}) 不随 (N) 改变；
* (K) 的典型元素保持 (O(1))；
* 有效离散 synaptic weight 是 (J^{\mathrm{eff}}=gK/N)。

若当前代码直接存储并更新有效矩阵 (W^{\mathrm{eff}})，同时动力学中没有显式除以 (N)，则应使用

[
\eta_N
======

\eta_{\mathrm{ref}}
\frac{N_{\mathrm{ref}}}{N},
]

并且 weight initialization、weight bounds 和单步更新都乘同样的 (N_{\mathrm{ref}}/N)。只缩放 recurrent current、却不缩放有效权重更新，会使不同 (N) 下的 learning timescale 和 equilibrium weight magnitude 不一致。

Vafidis 使用的 low-pass filtered prospective update，其时间常数应保持物理时间不变，不随 (N) 调整。训练时长也应按相同的运动时间、相同的完整环绕次数或相同的 angular occupancy 比较，而不是按总 synapse updates 比较。

## 5. (1/N) 和 (1/\sqrt N) 应分别用在哪里

建议采用下面的规则。

| 模型成分                                  |                             推荐标度 | 原因                                |
| ------------------------------------- | -------------------------------: | --------------------------------- |
| structured HD recurrent input         |              (1/N_{\mathrm{HD}}) | 保持 continuum integral 与总输入 (O(1)) |
| dense HR(\to)HD input                 |              (1/N_{\mathrm{HR}}) | 同上                                |
| one-to-one HD(\to)HR input            |                              (1) | 每个 neuron 只有固定数量输入                |
| global inhibition                     | population mean (N^{-1}\sum r_j) | 保持 mean mode 控制不变                 |
| direct GP visual current              |                              (1) | 固定 per-neuron sensory drive       |
| direct independent OU current noise   |                              (1) | 固定单神经元噪声强度                        |
| common angular cue jitter             |                              (1) | 固定 sensory uncertainty，单位为 radian |
| centered random connectivity disorder |                      (1/\sqrt N) | 保持随机矩阵谱半径和输入方差 (O(1))             |
| overlap/readout                       |                            (1/N) | 形成 intensive population statistic |

特别需要避免：

[
J_{ij}^{\mathrm{rand}}
\sim\frac1{\sqrt N}
]

但 (J_{ij}^{\mathrm{rand}}) 具有正的非零均值。此时平均 recurrent current 会按 (\sqrt N) 发散。(1/\sqrt N) 只适用于零均值或 E–I balanced 的随机部分，例如

[
J_{ij}
======

\frac{g_s}{N}K_{ij}
+
\frac{g_d}{\sqrt N}\xi_{ij},
\qquad
\mathbb E[\xi_{ij}]=0.
]

你的 Vafidis weights 若被限制为非负，就不应简单用 (1/\sqrt N) 初始化整个矩阵；更合适的是对正的 structured kernel 使用 (1/N)，把额外 disorder 单独定义成零均值项。

## 6. 噪声扫描应改用无量纲强度

此前的 `neuron count × noise std` 扫描中，raw std 很可能没有对应相同的 signal-to-noise ratio。建议定义

[
\tilde\sigma_{\mathrm{noise}}
=============================

\frac{
\sigma_{\mathrm{noise}}
}{
\operatorname{RMS}_{i,\theta}
\left[
I_i^{\mathrm{vis}}(\theta)
--------------------------

\langle I_i^{\mathrm{vis}}\rangle_\theta
\right]
}.
]

横轴使用 (\tilde\sigma_{\mathrm{noise}})，而不是绝对 current units。

若使用 OU process，应固定 stationary variance 和 correlation time：

[
d\xi_i
======

-\frac{\xi_i}{\tau_n}dt
+
\sqrt{\frac{2\sigma_n^2}{\tau_n}},dW_i.
]

其中 (\sigma_n) 是 stationary standard deviation，(\tau_n) 是 correlation time；两者均不随 (N) 改变。

还需要区分两种不同实验：

1. **Independent neural noise**：每个 neuron 有独立 (\xi_i(t))。增大 (N) 后 decoder error 自然可能下降，这是 population averaging 的真实效应。

2. **Sensory angular noise**：

[
\theta_{\mathrm{vis}}(t)=\theta(t)+\delta\theta(t),
]

所有 neuron 根据同一个 noisy heading 生成 cue。此时增加 (N) 不会自动消除 sensory uncertainty，更适合比较网络规模本身对学习和 attractor dynamics 的影响。

## 7. 异质 GP 表征下不要只使用普通 PVA

Clark 式 GP tuning curves可能多峰、非对称，并不满足 Vafidis classical ring 中“每个 neuron 有单一 preferred direction”的假设。因此普通 population vector average 会把 decoder mismatch 混进网络性能。

主 decoder 建议改为 normalized overlap：

[
m(\theta,t)
===========

\frac1N
\sum_{i=1}^N
q_i(\theta)r_i(t),
]

[
\hat\theta(t)
=============

\operatorname*{arg,max}_{\theta}
m(\theta,t),
]

并在所有 (N) 下使用同一个高分辨率 angular decoding grid。Clark 本身也是用 (1/N) normalized overlap 作为 population order parameter。

PVA 可以保留为辅助指标，用来显示“用 classical homogeneous decoder 读取 heterogeneous code 会损失多少性能”。

## 8. 推荐的最小修改顺序

第一步只改三处：

[
I_{\mathrm{rec}}=\frac{g_{\mathrm{rec}}}{N_{\mathrm{HD}}}K^{\mathrm{rec}}r^{\mathrm{HD}},
]

[
I_{\mathrm{HR}}=\frac{g_{\mathrm{HR}}}{N_{\mathrm{HR}}}K^{\mathrm{HR}}r^{\mathrm{HR}},
]

[
I_{\mathrm{mean}}=c(r_0-\bar r).
]

第二步把 GP cue 改成固定 baseline 与固定 modulation RMS，并把宽度参数统一为 radian。

第三步把学习变量改成 intensive (K)，保持 (\eta) 不随 (N) 变化；若继续更新 effective (W)，则令 (\eta\propto1/N)。

第四步将噪声拆为 independent current noise、shared angular noise 和 connectivity disorder，分别采用 (1)、(1) 和 (1/\sqrt N) 标度。

第五步使用 overlap decoder，并记录各通路的实际 current RMS：

[
\operatorname{RMS}(I_{\mathrm{vis}}),\quad
\operatorname{RMS}(I_{\mathrm{rec}}),\quad
\operatorname{RMS}(I_{\mathrm{HR}}),\quad
\operatorname{RMS}(e).
]

在训练前，这四个量应当在不同 (N) 下大致 collapse；否则后续的 performance comparison 仍然不能解释为纯粹的 finite-size effect。

最终最适合作为主模型的设定是：

[
\boxed{
\begin{aligned}
&\text{structured dense connectivity}: 1/N,\
&\text{GP visual cue}: \text{fixed per-neuron RMS},\
&\text{independent current noise}: \text{fixed per-neuron std},\
&\text{random zero-mean connectivity disorder}:1/\sqrt N,\
&\text{learning}: \text{update intensive kernels},\
&\text{decoder and inhibition}: \text{population means}.
\end{aligned}}
]

在这一正规化之后，若 (N=32) 的中等噪声仍然显著改善 performance，才更有资格解释为 noise-assisted depinning、stochastic resonance，或者噪声帮助网络逃离有限规模形成的离散稳定点，而不是简单的输入增益随 neuron count 改变。


---

总体判断：**这份设计文件的主线可行，Phase 0–5 的拆分也很合理，但还不建议原样直接进入 agent coding。** 其中有两个会直接改变训练时间尺度的关键缺口，以及几个会影响 (N\to\infty) 结论解释的问题。把下面四项“阻断性修改”补进设计文件后，就可以开始小步实现。

我目前检查的是设计 notebook 本身及其与 Clark、Vafidis 数学约定的一致性；由于这次没有看到 notebook 中列出的 `hd_dynamics.py`、`predictive_local.py` 等实际源码，所以无法确认文件路径、变量方向和当前 learning-rate 单位是否完全吻合。

[查看这份设计文件](sandbox:/mnt/data/population_scaling_revision_proposal%281%29.ipynb)

## 一、最关键的缺口：必须保证参考规模下的新旧模型等价

第 5–6 节提出

[
D=\frac{g}{N}Kr,
\qquad
\Delta K=\eta_K eP,\Delta t,
]

方向是对的，但“(g) 初始可设为 1、初始化和学习率保持 (O(1))”不足以指导迁移。

当前旧模型如果是

[
D_{\mathrm{old}}=W_{\mathrm{old}}r,
\qquad
\Delta W_{\mathrm{old}}
=======================

\eta_W eP,\Delta t,
]

那么在参考规模 (N_{\mathrm{ref}}) 下，要让新模型与旧模型逐时间步一致，必须满足

[
\frac{g}{N_{\mathrm{ref}}}K_0=W_{\mathrm{old}},
]

[
\frac{g}{N_{\mathrm{ref}}}\eta_K=\eta_W,
]

并且权重上下界也应满足

[
\frac{g}{N_{\mathrm{ref}}}K_{\max}=W_{\max}.
]

否则，虽然公式在渐近意义上采用了 (1/N)，但参考模型的 recurrent drive 或有效学习速度可能立刻缩小 (N_{\mathrm{ref}}) 倍。

有两个等价实现方案。

方案 A 更适合兼容现有配置：

[
K=W_{\mathrm{old}},
\qquad
g=N_{\mathrm{ref}},
\qquad
\eta_K=\eta_W.
]

于是

[
D_N
===

\frac{N_{\mathrm{ref}}}{N}Kr.
]

在 (N=N_{\mathrm{ref}}) 时，新旧模型完全一致，原来的初始化、weight bounds 和 learning rate 都可以保留。

方案 B 是更“纯粹”的 intensive-kernel 参数化：

[
g=1,
\qquad
K=N_{\mathrm{ref}}W_{\mathrm{old}},
\qquad
\eta_K=N_{\mathrm{ref}}\eta_W.
]

两者数学上等价。我更推荐方案 A 作为第一次实现，因为它最容易做 regression test，也不会立刻改变现有超参数的数值含义。

因此，第 13 节必须增加一个最高优先级测试：

> 在 (N=N_{\mathrm{ref}})、相同 seed、相同状态和相同输入下，旧模型与 scale-enabled 模型的每步 (I^{HH})、(I^{RH})、prediction error、weight update 和 firing rate 应在浮点误差范围内一致。

这个测试通过之前，不应运行 multi-(N) 实验。

## 二、HR 两翼最好分别归一化，而不是直接除以总 (N_{\mathrm{HR}})

设计文件第 5 节建议将左右两翼合并后使用

[
\frac{1}{N_{\mathrm{HR}}}
\sum_{k\in L\cup R}K_{ik}r_k.
]

这在量纲上没有错误，但会隐含一个额外的 (1/2)。Vafidis 原模型中左右 HR 是两个独立的速度通路，distal input 由 recurrent、right-rotation 和 left-rotation 三个独立求和项构成。其网络共有 (N_{\mathrm{HD}}=60)、(N_{\mathrm{HR}}=60)，其中 HD 每个方向两个细胞，HR 总体再分成左右两个各 30 个细胞的 population。 

更清楚的写法是

[
D_i^{RH}
========

\frac{g_R}{N_R}
\sum_{k\in R}K^R_{ik}r^R_k
+
\frac{g_L}{N_L}
\sum_{k\in L}K^L_{ik}r^L_k.
]

对称模型中令

[
N_R=N_L=N_{\mathrm{HD}}/2,
\qquad
g_R=g_L.
]

这样每一翼都是一个独立的 (O(1)) pathway，后续也更容易诊断左转和右转 gain 是否对称。若代码内部坚持存储一个合并的 (K^{RH})，也可以除以总 (N_{\mathrm{HR}})，但必须显式说明此时 pathway gain 比“逐翼平均”大两倍，不能让这个因子隐式存在。

参考规模兼容方案下，应分别使用

[
g_R=N_{R,\mathrm{ref}},
\qquad
g_L=N_{L,\mathrm{ref}}.
]

## 三、不能用简化公式重写 Vafidis plasticity pipeline

第 6 节写成

[
\Delta K_{ij}
=============

\eta e_i\bar r_j\Delta t
]

容易让 agent 把当前学习规则重写成 instantaneous outer product。Vafidis 实际规则的 presynaptic quantity 是经过突触和学习滤波的数据流，prospective weight change 也会经过低通处理；核心误差为 proximal firing 与 distal prediction 之间的差异。

建议把第 6 节改为：

[
\Delta K_{ij}
=============

\mathcal U
\left(
e_i,,
P^{\mathrm{pre}}*j,,
\text{existing filter states};
\eta_K,\tau*\delta,\Delta t
\right),
]

其中 (\mathcal U) 表示**当前代码已经实现的完整更新算子**。本轮只做两件事：

1. forward current 从 (Kr) 改为 (gKr/N_{\mathrm{pre}})；
2. 按参考规模映射 (g,\eta_K,K_{\max})。

不要更改 error definition、PSP filtering、prospective-update filtering、clipping 或时间积分顺序。设计文件中的 (\bar r_j) 也建议改名为 `pre_trace_j` 或 (P_j^{\mathrm{pre}})，避免被误读为 population mean。

“更新 intensive (K) 时不再额外除以 (N)”这个结论本身是正确的，因为

[
\frac{dD_i}{dt}
===============

\frac{g}{N}\sum_j
\frac{dK_{ij}}{dt}r_j
]

在相关的 population signal 下，求和给出 (O(N))，因此 pathway-level learning speed 可保持 (O(1))。但其数值学习率仍须通过上一节的参考等价关系确定。

## 四、必须先固定 (N) 的唯一语义

设计文件已经注意到 `n_theta` 的歧义，但这不能留到实现时再决定。Vafidis 原模型中：

[
N_{\mathrm{HD}}=60,
\qquad
N_{\mathrm{HR}}=60,
]

每个方向有两个 HD cells，所以 unique heading bins 为

[
N_{\mathrm{heading}}=N_{\mathrm{HD}}/2=30.
]

HR 总体分为左右两个 population，每一翼也有 30 个细胞。

建议 schema 直接使用四个无歧义量：

```text
n_hd_cells
n_heading_bins = n_hd_cells // 2
n_hr_cells
n_hr_per_wing = n_hr_cells // 2
```

并设置硬约束：

```text
n_hd_cells % 2 == 0
n_hr_cells % 2 == 0
n_heading_bins == n_hd_cells // 2
n_hr_per_wing == n_hr_cells // 2
```

第一轮最好继续维持

[
N_{\mathrm{HR}}=N_{\mathrm{HD}},
]

不要同时扫描 HD/HR population ratio。否则 (N) 的变化又会和 architecture ratio 混杂。

## 五、Phase 1 的设计是正确的，但不要把“性能单调改善”设为验收条件

设计文件第 1、10、13 节的科学目标基本正确。引入 (1/N) 后应期待的是：

[
\text{mean intensive observables}\rightarrow O(1)\text{ limit},
]

[
\text{finite-size fluctuations}\rightarrow 0,
]

而不是保证

[
\mathrm{RMSE}(N_1)>\mathrm{RMSE}(N_2)>\mathrm{RMSE}(N_3)
]

严格单调成立。

更准确的假设应当是：

* pathway current、population rate、bump width 和有效 gain 随 (N) 收敛到平台；
* seed-to-seed variance 随 (N) 减小；
* GP population sampling imbalance 通常呈 (N^{-1/2}) 量级减小；
* discretization-induced pinning、dead zone 或 endpoint quantization 可能减弱；
* 平均 RMSE 可能下降、趋于平台，也可能存在有限 (N) 最优点。

因此，最后不能把“较大的 (N) 没有更好”直接判定为 scale implementation 失败。真正的失败信号是：normalized recurrent current、学习速度、rate 工作点或有效 Jacobian 随 (N) 系统性漂移。

Clark 的 low-rank 写法本身就是

[
J=\frac1N UV^\top,
\qquad
\kappa=\frac1N V^\top\phi,
]

目的是得到良好的 large-(N) population dynamics；这支持设计文件采用 intensive population quantities。

## 六、建议新增“冻结 kernel 的离散化控制”

目前设计直接比较“每个 (N) 都重新训练”的模型。这样若结果不同，很难区分：

1. forward discretization 的有限规模误差；
2. local learning 在不同 (N) 下得到不同解；
3. GP population sample 的不同；
4. decoder 的有限样本误差。

建议在 Phase 1 加一个非常有价值的 Phase 1A：

先构造一个固定、平滑的 continuum-like kernel，或从一个已训练网络得到 circularly averaged kernel (K(\Delta\theta))，然后在不同 (N) 上离散化：

[
K^{(N)}_{ij}
============

K(\theta_i-\theta_j).
]

冻结权重，不进行学习，只测试：

* static bump maintenance；
* velocity gain；
* endpoint map；
* pinning；
* effective current；
* full-state Jacobian。

如果这个控制都不能随 (N) 收敛，说明 forward scaling 或角度离散化仍有错误。只有它通过后，才进入“不同 (N) 各自在线学习”的 Phase 1B。

## 七、GP population 必须使用 nested sampling

这是设计文件目前遗漏的最重要实验控制之一。

若 (N=120,240,360) 各自独立重新抽取 GP profiles，那么不同 (N) 的差异同时包含“population size”和“随机小鼠身份”。应当对每个 mouse seed 先生成一个最大规模的 profile bank：

[
{q_i(\theta)}*{i=1}^{N*{\max}},
]

然后让小网络使用其嵌套子集，例如：

[
\mathcal P_{120}\subset
\mathcal P_{240}\subset
\mathcal P_{360}.
]

最好采用随机排列后的前 (N) 个，而不是按某个 tuning statistic 选择。不同 mouse seeds 对应不同 master banks。

运动轨迹、shared angular noise 和训练/测试切分也应在不同 (N) 间使用 common random numbers。这样 paired comparison 才能直接回答“增加 neuron count 做了什么”。

还应额外运行 independent-resampling condition，用来估计 population realization 本身的方差。

## 八、matched-drive 是有效控制，但不是纯粹的“幅值归一化”

第 7 节提出

[
\delta q_i(\theta)
==================

q_i(\theta)-\mu_N(\theta)
]

会把每个角度上的 population common mode 精确投影掉：

[
\frac1N\sum_i\delta q_i(\theta)=0.
]

这不仅改变 amplitude，还改变了 teacher covariance 的 uniform mode。因此它是一个有价值的 symmetry-controlled benchmark，但不是原始 GP cue 的简单重标度。

建议保留三个明确命名的 condition：

[
\texttt{raw_gp}:
\quad
I_i=I_0+A(q_i-1),
]

[
\texttt{rms_matched_gp}:
\quad
I_i=I_0+A\frac{q_i-1}{s_q},
]

[
\texttt{common_mode_removed_gp}:
\quad
I_i=I_0+A\frac{q_i-\mu_N(\theta)}{s_q}.
]

第二个只匹配 modulation RMS，仍保留有限规模 common-mode fluctuation；第三个才是设计文件当前所谓的 matched-drive。这样可以判断性能改善究竟来自 gain matching，还是来自人为删除 common mode。

Phase 1 不应改变当前 visual cue。这个修改应严格留在 Phase 2，设计文件在这一点上是正确的。

## 九、pair-shared 与 pair-independent 不是简单的“主条件/扰动条件”

pair-shared 更接近原始 Vafidis architecture：两个具有相同 heading preference 的 HD cells 接受相同视觉 bump，并分别投射到左右 HR wing。

但 Clark generator 的核心是对神经元独立抽取 heterogeneous tuning functions；强制一对细胞共享 profile 会引入人工的完美相关，并把独立样本数从 (N_{\mathrm{HD}}) 降到

[
N_{\mathrm{profile}}=N_{\mathrm{HD}}/2.
]

因此：

* `pair_shared` 是 **Vafidis-architecture-faithful**；
* `pair_independent` 是 **Clark-population-faithful**；
* 两者回答不同问题。

在你的课题主线“Clark heterogeneity + Vafidis learning”下，我不建议预先把 pair-independent 定义成不自然的扰动。Phase 2 应把它们作为两个并列模型族，并分别报告 large-(N) scaling。尤其在拟合 (N^{-1/2}) 时，要明确这里的 (N) 是实际 HD cells，还是 independent GP profiles。

## 十、Jacobian 验收需要写得更精确

设计文件提到“有效谱”和“近中性 tangent mode”，但要防止 agent 只计算 (W^{\mathrm{eff}}) 的 eigenvalues。

这个模型包含：

* HD proximal voltage；
* HD distal voltage/current；
* synaptic filter states；
* HR states；
* 可能的 learning traces。

冻结权重后的 neural-state Jacobian 是一个 block Jacobian。连续吸引子的 tangent mode 应在这个**完整 frozen-weight dynamical Jacobian**中寻找，而不是直接在 (W^{\mathrm{eff}}) 中寻找。

应分别输出：

```text
connectivity spectrum:
    eig(W_eff_HH)
    singular values / spectral radius of pathway matrices

neural dynamical spectrum:
    eig(J_full_state)

learning dynamics diagnostics:
    weight-update norm and error decay
```

Clark 的 connectivity degeneracy、neural-state Jacobian 和 Vafidis learning dynamics 不是同一个谱对象。Clark 的不同 settings 也分别使用了 data-derived softplus、normalized-softplus generator 和 large-(N) erf dynamics，设计文件将它们分开处理是准确的。 

## 十一、首轮 (N) 选择建议调整

(120,240,360) 可用于工程 smoke test，但只有三个且并非对数间隔，不足以支持“趋近 (N\to\infty)”的有限规模拟合。

建议 pilot 使用：

[
N_{\mathrm{HD}}
\in
{60,120,240,480},
]

对应 unique heading bins：

[
{30,60,120,240}.
]

先做 5–10 个 nested master seeds。然后针对可承受的规模做 30-mouse 统计。可以拟合：

[
O(N)=O_\infty+aN^{-\alpha},
]

其中不要预先强制所有指标都使用 (\alpha=1/2)。population fluctuation 的标准差可能接近 (N^{-1/2})，variance 接近 (N^{-1})，而 discretization bias 或 pinning barrier 未必服从相同指数。

## 最终审批结论

这份 notebook 可以作为 agent coding 的基础，但我建议先让 Codex把以下内容补入设计文件，再开始修改代码：

1. 明确参考规模等价映射：

[
K_0=\frac{N_{\mathrm{ref}}}{g}W_0,\quad
\eta_K=\frac{N_{\mathrm{ref}}}{g}\eta_W,\quad
K_{\max}=\frac{N_{\mathrm{ref}}}{g}W_{\max}.
]

2. HR 左右两翼分别按各自 population size 归一化。

3. 明确本轮不得改写现有 predictive-learning pipeline，只改 forward scaling 和参数映射。

4. 固定 `n_hd_cells`、`n_heading_bins`、`n_hr_cells`、`n_hr_per_wing` 的语义。

5. 增加 (N=N_{\mathrm{ref}}) 新旧逐步等价测试。

6. 增加 frozen-kernel discretization control。

7. 增加 nested GP population 和 common-random-number 实验协议。

8. 将成功目标从“性能随 (N) 必然单调提高”改为“intensive dynamics 收敛，有限规模方差与 pinning 减弱”。

完成这些修改后，**可以开始 Phase 0 与 Phase 1 的 agent coding**；softplus、common-mode removal、pair sharing 和 noise rescaling 都继续留在后续独立阶段，不要首个 patch 一次性实现。
