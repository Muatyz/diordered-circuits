# Block-multirate data stream for Vafidis predictive learning

> 本文档用于说明当前 `model-dev/learning` 中的 block-multirate 训练实现，重点回答两个问题：一个 block 内各物理量究竟按什么顺序更新，以及为什么这种更新在当前模型中是一个可控、可验证的近似。单时钟基线及符号定义见 [Vafidis.md](Vafidis.md#one-timestep-data-stream)。

## 汇报摘要

当前实现没有把神经网络的时间步从 $0.1\,\mathrm{ms}$ 放大到 $10\,\mathrm{ms}$，也没有改变 Vafidis learning rule。它使用两个数值时钟：

- neural microstep：$\Delta t=0.0001\,\mathrm{s}=0.1\,\mathrm{ms}$；
- plasticity block：$T_B=0.01\,\mathrm{s}=10\,\mathrm{ms}$，因此每个完整 block 含

  $$
  K=\frac{T_B}{\Delta t}=100
  $$

  个 neural microsteps。

每个 microstep 仍按原有固定顺序更新真实角度、视觉/速度输入、HD/HR dynamics、两级 PSP 和局部 prediction error。一个 block 内暂时固定两组可塑权重，但保留每一步的

$$
\vec E_{\mathrm{HD}},\qquad \vec P_{\mathrm{HD}},\qquad \vec P_{\mathrm{HR}}.
$$

到 block 边界后，代码对原逐步 Euler plasticity recurrence 作闭式复合，一次得到 block 末端的 $\boldsymbol\delta$ 和 $\mathbf W$。给定同一组 $E/P$ samples，这一步与逐 microstep 更新 plasticity 的代数结果相同；完整耦合系统中唯一的 multirate 近似，是新权重在 block 内不立即反馈到后续神经动力学。

当前实验配置文件仍把 `single_clock` 保留为科学基线，但仓库的 `work/dev.bat` 和 `work/release.bat` 会加载 `configs/profiles/block_multirate.yaml`，所以通过这两个快捷命令启动的训练使用本文所述的 10 ms block data stream。

## 1. 为什么不能直接使用一个 10 ms timestep

当前模型包含明显分离、但并非完全解耦的时间尺度：

| 状态或过程 | 当前时间尺度 | block 模式中的处理 |
| --- | ---: | --- |
| HD proximal fixed-drive mode | $C/(g_L+g_D)=1/3\,\mathrm{ms}$ | 每 $0.1\,\mathrm{ms}$ 更新；当前配置对该线性子步使用 `exact_linear` |
| HD distal voltage | $\tau_{l,\mathrm{HD}}=10\,\mathrm{ms}$ | 每 $0.1\,\mathrm{ms}$ 更新 |
| distal current / PSP first stage | $\tau_s=65\,\mathrm{ms}$ | 每 $0.1\,\mathrm{ms}$ 更新 |
| HD-to-HR low-pass | $\tau_{\mathrm{HD}\to\mathrm{HR}}=65\,\mathrm{ms}$ | 每 $0.1\,\mathrm{ms}$ 更新 |
| plasticity induction | $\tau_\delta=100\,\mathrm{ms}$ | 每个 sample 的作用被闭式累计 |
| weights | $\dot W=\eta\delta$，没有固定的 $\tau_W$ | block 内固定，边界时一次提交 |

最快的 proximal mode 远小于 10 ms，所以直接用 10 ms 推进整个网络会漏掉甚至破坏快动力学。block-multirate 的含义不是“粗化所有状态”，而是保留神经微步，只降低慢可塑性矩阵更新及约束检查的提交频率。

## 2. 两层状态和两个时钟

令 $q$ 表示 block index，$k=0,\ldots,K$ 表示 block 内 microstep index。把状态分为：

1. neural state

   $$
   X_{q,k}=\left(
   \theta,
   \vec I_{\mathrm{HD},d},
   \vec V_{\mathrm{HD},d},
   \vec V_{\mathrm{HD},a},
   \vec P_{\mathrm{syn}},
   \vec P,
   \vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP},
   \vec I_{\mathrm{HR}},
   \vec r_{\mathrm{HR}},
   \vec r_{\mathrm{HD}}
   \right);
   $$

2. plastic state

   $$
   Y_q=\left(
   \boldsymbol\delta_{\mathrm{HD}\to\mathrm{HD}}^q,
   \boldsymbol\delta_{\mathrm{HR}\to\mathrm{HD}}^q,
   \mathbf W_{\mathrm{HD}\to\mathrm{HD}}^q,
   \mathbf W_{\mathrm{HR}\to\mathrm{HD}}^q
   \right).
   $$

一个 block 的抽象数据流是

$$
X_{q,k+1}
=F_{\Delta t}\!\left(X_{q,k};\mathbf W^q,u_{q,k+1}\right),
\qquad k=0,\ldots,K-1,
$$

$$
(\vec E_{q,k+1},\vec P_{q,k+1})=H(X_{q,k+1}),
$$

然后在 block 边界执行

$$
Y_{q+1}
=\Phi_K\!\left(Y_q;\{\vec E_{q,j},\vec P_{q,j}\}_{j=1}^{K}\right).
$$

$F_{\Delta t}$ 是原来的 ordered neural step，$\Phi_K$ 是原 Euler plasticity recurrence 的闭式复合。更新后的 $\mathbf W^{q+1}$ 从下一个 block 的第一个 microstep 起参与 distal drive；不会追溯性地重新计算刚结束的 block。

## 3. 一个 block 内的 microstep data stream

下列步骤在每个 microstep 都执行。为简化记号，本节省略 block index $q$，但在整个 block 内使用同一组 $\mathbf W^q$。

### 3.1 生成速度并更新真实角度

本步速度由 constant 或 OU driver 产生：

$$
v^{k+1}=
\begin{cases}
v_{\mathrm{const}}, & \text{constant},\\[2mm]
v^k+\dfrac{\Delta t}{\tau_v}(\mu_v-v^k)
+\sigma_v\sqrt{\dfrac{2\Delta t}{\tau_v}}\,\xi^{k+1},
& \text{OU},
\end{cases}
$$

其中 $\xi^{k+1}\sim\mathcal N(0,1)$。随后更新

$$
\theta^{k+1}=\operatorname{wrap}\!\left(\theta^k+v^{k+1}\Delta t\right).
$$

block 模式不改变 RNG 的调用频率：每个 neural microstep 仍消费一个与 single-clock 对应的速度 sample。

### 3.2 构造 visual 和 velocity inputs

对第 $i$ 个 HD neuron，令

$$
\Delta\theta_i^{k+1}
=\operatorname{wrap}(\theta_{\mathrm{HD},i}-\theta^{k+1}).
$$

visual teacher 开启时，proximal current 为

$$
I_{\mathrm{HD},a,i}^{k+1}
=s_{\mathrm{prox}}
\left[
A_{\mathrm{vis}}
e^{\kappa_{\mathrm{vis}}(\cos\Delta\theta_i^{k+1}-1)}
-b_{\mathrm{vis}}+I_{\mathrm{exc}}
\right],
$$

关闭时为零；显式配置的噪声在对应输入位置加入。HR velocity input 为

$$
\vec I_{\mathrm{vel}\to\mathrm{HR}}^{k+1}
=
\begin{bmatrix}
+k_vv^{k+1}\vec 1_{N_{\mathrm{HR}}/2}\\
-k_vv^{k+1}\vec 1_{N_{\mathrm{HR}}/2}
\end{bmatrix}.
$$

### 3.3 用 block-frozen weights 和旧 rates 计算 HD distal drive

$$
\vec I_{\mathrm{HD}\leftarrow\mathrm{HD}}^{k}
=\mathbf W_{\mathrm{HD}\to\mathrm{HD}}^{q}\vec r_{\mathrm{HD}}^k,
$$

$$
\vec I_{\mathrm{HD}\leftarrow\mathrm{HR}}^{k}
=\mathbf W_{\mathrm{HR}\to\mathrm{HD}}^{q}\vec r_{\mathrm{HR}}^k,
$$

$$
\vec D_{\mathrm{HD},d}^{k}
=\vec I_{\mathrm{HD}\leftarrow\mathrm{HD}}^{k}
+\vec I_{\mathrm{HD}\leftarrow\mathrm{HR}}^{k}
-b_{\mathrm{HD}}\vec 1.
$$

与 single-clock 一样，本步 distal drive 不读取本步稍后才生成的新 HD/HR rates。

### 3.4 更新 HD distal current 和 distal voltage

先更新 current：

$$
\vec I_{\mathrm{HD},d}^{k+1}
=\vec I_{\mathrm{HD},d}^{k}
+\frac{\Delta t}{\tau_s}
\left(-\vec I_{\mathrm{HD},d}^{k}+\vec D_{\mathrm{HD},d}^{k}\right),
$$

再让 voltage 读取刚更新的 current：

$$
\vec V_{\mathrm{HD},d}^{k+1}
=\vec V_{\mathrm{HD},d}^{k}
+\frac{\Delta t}{\tau_{l,\mathrm{HD}}}
\left(-\vec V_{\mathrm{HD},d}^{k}
+\vec I_{\mathrm{HD},d}^{k+1}\right).
$$

### 3.5 更新两级 presynaptic PSP traces

对 $X\in\{\mathrm{HD},\mathrm{HR}\}$，第一级读取旧 presynaptic rate：

$$
\vec P_{X,\mathrm{syn}}^{k+1}
=\vec P_{X,\mathrm{syn}}^{k}
+\frac{\Delta t}{\tau_s}
\left(-\vec P_{X,\mathrm{syn}}^{k}+\vec r_X^k\right),
$$

第二级读取刚更新的 synaptic stage：

$$
\vec P_X^{k+1}
=\vec P_X^k
+\frac{\Delta t}{\tau_{l,\mathrm{HD}}}
\left(-\vec P_X^k+\vec P_{X,\mathrm{syn}}^{k+1}\right).
$$

### 3.6 更新 HD-to-HR delay、HR current 和 HR rate

$$
\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,k+1}
=\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,k}
+\frac{\Delta t}{\tau_{\mathrm{HD}\to\mathrm{HR}}}
\left(-\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,k}
+\vec r_{\mathrm{HD}}^k\right),
$$

$$
\vec I_{\mathrm{HR}}^{k+1}
=\mathbf W_{\mathrm{HD}\to\mathrm{HR}}
\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,k+1}
+\vec I_{\mathrm{vel}\to\mathrm{HR}}^{k+1}
-b_{\mathrm{HR}}\vec1,
$$

$$
\vec r_{\mathrm{HR}}^{k+1}=f(\vec I_{\mathrm{HR}}^{k+1}).
$$

$\mathbf W_{\mathrm{HD}\to\mathrm{HR}}$ 固定且不可塑；新 HR rate 从下一个 microstep 起参与 HD distal drive。

### 3.7 更新 HD proximal voltage

当前主配置使用 `exact_linear` 更新 Vafidis Eq. (4) 的线性 proximal 子问题。令

$$
g_\Sigma=g_L+g_D,
\qquad
\lambda_a=\frac{g_\Sigma}{C},
$$

并在本子步内把已经更新的 $\vec V_{\mathrm{HD},d}^{k+1}$ 与 proximal current 视为分段常数，则

$$
\vec V_{\infty}^{k+1}
=\frac{g_D\vec V_{\mathrm{HD},d}^{k+1}
+\vec I_{\mathrm{HD},a}^{k+1}}{g_\Sigma},
$$

$$
\vec V_{\mathrm{HD},a}^{k+1}
=\vec V_{\infty}^{k+1}
+\left(\vec V_{\mathrm{HD},a}^{k}-\vec V_{\infty}^{k+1}\right)
e^{-\lambda_a\Delta t}.
$$

这只对已经给定 drive 的 proximal 线性子步精确，并不声称整个耦合网络被解析求解。若配置选择 `forward_euler`，则此处回到 `Vafidis.md` 中的显式 Euler 公式；其余 block data stream 不变。

### 3.8 计算 HD rate、distal prediction 和局部 error

$$
\vec V_{\mathrm{HD},ss}^{k+1}
=p_{d\to a}\vec V_{\mathrm{HD},d}^{k+1},
$$

$$
\vec r_{\mathrm{HD}}^{k+1}
=f(\vec V_{\mathrm{HD},a}^{k+1}),
\qquad
\vec r_{\mathrm{HD},d\text{-}\mathrm{pred}}^{k+1}
=f(\vec V_{\mathrm{HD},ss}^{k+1}),
$$

$$
\vec E_{\mathrm{HD}}^{k+1}
=\vec r_{\mathrm{HD}}^{k+1}
-\vec r_{\mathrm{HD},d\text{-}\mathrm{pred}}^{k+1}.
$$

### 3.9 保存局部 plasticity samples，但暂不提交权重

将本步的

$$
\left(
\vec E_{\mathrm{HD}}^{k+1},
\vec P_{\mathrm{HD}}^{k+1},
\vec P_{\mathrm{HR}}^{k+1}
\right)
$$

依次写入 block buffer。内存中的 $\boldsymbol\delta^q$ 和 $\mathbf W^q$ 在 block 内保持不变；不过这不表示数学上的 $\delta$ 被忽略。其逐步 Euler 轨迹会在 block 边界由闭式公式完整重建。

新 neural states $X_{k+1}$ 立即成为下一个 microstep 的输入，因此被延迟的只有 plastic weights 对神经动力学的反馈。

## 4. Block 边界的 plasticity update

令一个实际 block 含 $M$ 个 samples。通常 $M=K=100$；训练结束或 checkpoint 落在 block 中间时允许 $1\le M<K$，并立即 flush partial block。

对 pathway $X\in\{\mathrm{HD},\mathrm{HR}\}$ 定义

$$
\boldsymbol\Pi_{X,j}
=\vec E_{\mathrm{HD},j}\vec P_{X,j}^{\top},
\qquad j=1,\ldots,M,
$$

以及

$$
h=\frac{\Delta t}{\tau_\delta},
\qquad
a=1-h.
$$

原 single-clock Euler recurrence 是

$$
\boldsymbol\delta_{X,j}
=a\boldsymbol\delta_{X,j-1}
+(1-a)\boldsymbol\Pi_{X,j},
$$

$$
\mathbf W_{X,j}
=\mathbf W_{X,j-1}
+\Delta t\,\eta_X\boldsymbol\delta_{X,j}.
$$

把这两个递推复合 $M$ 次，得到 block 末端 induction state：

$$
\boxed{
\boldsymbol\delta_{X,M}
=a^M\boldsymbol\delta_{X,0}
+(1-a)\sum_{j=1}^{M}a^{M-j}\boldsymbol\Pi_{X,j}
}
$$

以及约束投影前的权重：

$$
\boxed{
\widetilde{\mathbf W}_{X,M}
=\mathbf W_{X,0}
+\Delta t\,\eta_X
\left[
\left(\sum_{m=1}^{M}a^m\right)\boldsymbol\delta_{X,0}
+\sum_{j=1}^{M}
\left(1-a^{M-j+1}\right)\boldsymbol\Pi_{X,j}
\right]
}
$$

最后应用配置中显式启用的约束：

$$
\mathbf W_{X}^{q+1}
=\mathcal C_X\!\left(\widetilde{\mathbf W}_{X,M}\right),
\qquad
\boldsymbol\delta_X^{q+1}=\boldsymbol\delta_{X,M}.
$$

baseline 没有 clipping、symmetry、balance 或 diagonal projection，所以 $\mathcal C_X$ 在基线中是恒等映射。

### 4.1 实现没有逐个构造所有 outer products

把 block samples 排成

$$
\mathbf E\in\mathbb R^{M\times N_{\mathrm{HD}}},
\qquad
\mathbf P_X\in\mathbb R^{M\times N_X},
$$

任意加权 outer-product sum 都可写成

$$
\sum_{j=1}^{M}c_j
\vec E_j\vec P_{X,j}^{\top}
=\mathbf E^{\top}\operatorname{diag}(\vec c)\mathbf P_X.
$$

代码实际计算

$$
\mathbf E^{\top}(\vec c\odot\mathbf P_X),
$$

由密集矩阵乘法一次完成整个 block 的累计。这保留了 sample-by-sample 的 $E_jP_j^\top$ correlation，同时减少每步 Python/NumPy 层面的 outer product、矩阵加法、约束和完整状态检查。

特别地，代码没有使用

$$
\operatorname{outer}(\operatorname{mean}E,\operatorname{mean}P),
$$

因为一般而言

$$
\operatorname{mean}(EP^\top)
\ne
\operatorname{mean}(E)\operatorname{mean}(P)^\top.
$$

后一种写法会丢失 block 内 error 与 presynaptic PSP 的时间相关性，并真正改变局部学习信号。

## 5. Single-clock 与 block-multirate 的对应关系

| 项目 | `single_clock` | `block_multirate` |
| --- | --- | --- |
| neural timestep | $\Delta t$ | $\Delta t$，不变 |
| velocity/noise samples | 每步生成 | 每个 microstep 生成，同频率 |
| compartment、HR、PSP | 每步 ordered update | 每个 microstep 相同 ordered update |
| $E$ 和 $P$ | 每步计算 | 每个 microstep 计算并保存 |
| $\delta$ | 每步显式 materialize | block 边界闭式得到相同 Euler 末态 |
| $W$ 的累计 | 每步 $W\leftarrow W+\Delta t\eta\delta$ | block 边界闭式累计所有 virtual microstep increments |
| 新 $W$ 对神经状态的反馈 | 下一 microstep | 下一 block |
| constraints | 每步 | block 边界 |
| state validation | 每步 | block/checkpoint 边界 |

当 $K=1$ 时，两种方法退化为同一个完整训练 data stream，测试要求状态与权重在约 $10^{-13}$ tolerance 内一致。

## 6. 为什么这种方法是合理的

### 6.1 它保持了 Vafidis learning rule

当前模型的可塑性方程是

$$
\tau_\delta\dot{\boldsymbol\delta}
=-\boldsymbol\delta+\vec E\vec P^\top,
\qquad
\dot{\mathbf W}=\eta\boldsymbol\delta.
$$

权重方程中没有 $-\mathbf W$。block 实现只是复合现有 Euler recurrence，没有添加 weight decay、全局 loss、backpropagation、future information 或导师草图中可能出现的 $\tau_W\dot W=-W+\cdots$ 项。

### 6.2 给定 local samples 时，plasticity integration 是代数精确的

$\delta$ subsystem 对给定 $\Pi_j=E_jP_j^\top$ 是线性 affine recurrence，$W$ 又是 $\delta$ 的离散积分。因此可以解析地复合一个 block，而不需要用 block mean 近似 local drive。

这意味着误差不来自“漏算了 99 个 learning samples”，也不来自“把 $\tau_\delta$ 的 timestep 改成 10 ms”。给定同一组 $E/P$ samples，block 公式与逐步 Euler plasticity update 一致。

### 6.3 被近似的只有 block 内 weight feedback

$\boldsymbol\delta$ 不直接进入 neural dynamics，只有 $\mathbf W$ 通过下一步 distal drive 反馈。因此延迟 materialize $\delta$ 本身不会改变 block 内神经轨迹；真正的 splitting error 仅来自 single-clock 本会使用

$$
\mathbf W_1,\mathbf W_2,\ldots,\mathbf W_{M-1},
$$

而 block 模式一直使用 $\mathbf W_0$。

这个近似是否足够好，不能仅凭“$\tau_\delta=100\,\mathrm{ms}$”下结论，因为 $W$ 没有固定 intrinsic time constant。更直接的 a posteriori 判据是每个 block 的相对权重变化：

$$
\epsilon_{W,X}^{(q)}
=\frac{\|\mathbf W_X^{q+1}-\mathbf W_X^q\|_F}
{\max(\|\mathbf W_X^q\|_F,\epsilon)}.
$$

当两个 pathway 都满足 $\epsilon_{W,X}^{(q)}\ll1$ 时，block 内固定 $W$ 对 distal drive 的影响是一阶小量。若训练早期或某组参数使该量不再小，应缩短 $T_B$，并重新与 single-clock 比较。

### 6.4 快动力学仍被高分辨率解析

block 模式没有用慢时钟替代 neural clock。最快 proximal subproblem 仍每 0.1 ms 更新，并使用其分段常 drive 下的 exact-linear map；distal、PSP 和 HR states 也仍逐 microstep 演化。因此 slow-fast approximation 只放在 slow weights 的反馈位置，而不是放在容易失稳的电压、电流或 firing-rate states 上。

### 6.5 它保持局部性和因果顺序

每个 synaptic update 只依赖当时的 postsynaptic HD error 与对应 presynaptic PSP。block 公式虽然批量计算，但没有引入全局 objective，也没有使用 block 之后的信息。sample 权重只来自已知的 $\Delta t/\tau_\delta$ 衰减核，因此是原 causal filter 的重排，不是新的 learning rule。

### 6.6 Checkpoint 和训练终点不会保存未提交的 plasticity

以下任一条件出现时，runner 都会 flush 当前 block：

- 收集满 $K$ 个 samples；
- 到达训练终点；
- 到达 weight snapshot 或 recovery checkpoint；
- 到达 behavioral checkpoint-selection 检查；
- 到达 early-stopping 检查。

因此 partial final block 不会丢失，保存的 checkpoint 也不会出现“history 已到时间 $t$，weights 却还停在更早 block 边界”的不一致。

## 7. 已有验证证据

当前测试和本地 benchmark 覆盖了不同层次的等价性：

1. **Plasticity algebra**：随机给定 $E/P$ sequence，block 公式与重复 single-clock plasticity update 在约 $2\times10^{-14}$ tolerance 内一致。
2. **$K=1$ 完整 data stream**：block size 为一个 microstep 时，HD/HR rates、voltages、$\delta$ 和 $W$ 在约 $10^{-13}$ tolerance 内一致。
3. **Partial block**：训练终点不整除 block size 时，最后一个 block 会被 flush，且权重/induction state 有限并已更新。
4. **Matched-noise short training**：现有自动测试要求 block 与 single-clock 的相对 weight error 小于 $10^{-4}$，HD-rate RMS difference 小于 $10^{-6}$。
5. **50,000-step workstation benchmark**：对 $N=60/120$，10 ms block 分别观察到约 $2.39\times/2.73\times$ speedup；HD recurrent weight 的相对 Frobenius difference 约为 $3.8\times10^{-7}/1.1\times10^{-6}$。

第 5 点是实现尺度的 matched-stream benchmark，不是长期科学等价性的最终证据。详细记录见 [2026-08-19 multirate integrator report](../model-dev/learning/reports/notes/2026-08-19_multirate_training_integrator.md)。

## 8. 合理性的边界与仍需验证的问题

以下内容不能由短时单元测试推出：

1. 60,000 s 长训练后 selected snapshot 是否与 single-clock 得到同样的 weight basin；
2. 多个随机种子下 absolute learning error、weight profile 和最佳训练时刻是否一致；
3. frozen constant-velocity PI、OU PI、depinning velocity 和 autonomous phase flow 是否保持在预设容差内；
4. 启用 clipping、symmetry、balance 或 diagonal constraint 时，block-boundary projection 是否足够接近逐步 projection。

第 4 点尤其重要：约束投影通常是非线性的，所以“给定 $E/P$ 时 block plasticity 代数精确”的结论只覆盖未约束 recurrence，或者把相同 projection 约定放在同一个比较边界的情形。任何 constrained experiment 都应单独做 block-size convergence。

建议长期验收至少比较

$$
T_B\in\{\Delta t,\,1\,\mathrm{ms},\,5\,\mathrm{ms},\,10\,\mathrm{ms}\},
$$

并使用相同初始化和相同分组件随机流，报告：

- $\|\Delta W\|_F/\|W\|_F$ 的 block-level distribution；
- absolute local learning error；
- HD-to-HD 与 HR-to-HD weight profiles；
- 各 snapshot 的 frozen PI error；
- 最终选择的 snapshot time 与 held-out PI 指标。

如果减小 $T_B$ 后这些量收敛，10 ms 才能被视为当前参数范围内经验证的最大高效 block，而不是仅凭时间尺度直觉选择的常数。

## 9. 配置、代码和测试映射

当前快捷训练命令等价于在主配置上叠加 profile：

```bat
python -m learning.experiments.run_vafidis_toy ^
  --config configs\experiments\vafidis_toy.yaml ^
  --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml ^
  --profile configs\profiles\block_multirate.yaml
```

相关实现位置：

- profile：[block_multirate.yaml](../model-dev/learning/configs/profiles/block_multirate.yaml)
- microstep ordered dynamics：[vafidis_toy.py](../model-dev/learning/src/learning/models/vafidis_toy.py)
- block buffer 与 flush logic：[run_vafidis_toy.py](../model-dev/learning/src/learning/experiments/run_vafidis_toy.py)
- closed-form plasticity composition：[predictive_local.py](../model-dev/learning/src/learning/plasticity/predictive_local.py)
- multirate integration tests：[test_multirate_training.py](../model-dev/learning/tests/test_multirate_training.py)
- algebraic parity test：[test_predictive_local_plasticity.py](../model-dev/learning/tests/test_predictive_local_plasticity.py)

## 10. 一句话向导师解释

> 我们没有用 10 ms 粗步长推进神经网络；神经元、电流、电压和 PSP 仍以 0.1 ms 更新。我们只在 10 ms 内暂时冻结缓慢变化的权重，完整保存每一步的局部 error/PSP correlation，然后对原 Euler plasticity recurrence 作解析的 block composition。因而 learning rule 和局部 samples 都被保留，唯一需要通过 block-size convergence 验证的近似，是新权重对同一 block 内神经动力学的延迟反馈。
