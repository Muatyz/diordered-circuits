# Vafidis predictive local learning

> 本笔记以当前 `src/learning` 和 release baseline `configs/experiments/vafidis_toy.yaml` 为准. 论文描述连续时间方程；当前代码按 release 的固定顺序, 用一个全局 $\Delta t$ 在线推进. 

- 数量
    - head-direction cells: $N_{\mathrm{HD}}=60$
    - head-rotation cells: $N_{\mathrm{HR}}=60$
    - $N_{\mathrm{LHR}}=N_{\mathrm{RHR}}=30$
    - 60 个 HD cell 两两共享 preferred direction, 因此只有 30 个 unique headings

- firing rate
    - $\vec r_{\mathrm{HD}}=f(\vec V_{\mathrm{HD},a})\in\mathbb R^{N_{\mathrm{HD}}}$
    - $\vec r_{\mathrm{HR}}=[\vec r_{\mathrm{LHR}};\vec r_{\mathrm{RHR}}]\in\mathbb R^{N_{\mathrm{HR}}}$
    - $f(x)=f_{\max}/[1+\exp(-\beta(x-x_{1/2}))]$
        - baseline 使用 $f_{\max}=0.15\,\mathrm{ms}^{-1}=150\,\mathrm{spikes/s}$；rate、PSP 和 prediction error 不做 unit-peak normalization

- connection
    - $\mathbf W_{\mathrm{HD}\to\mathrm{HD}}\in\mathbb R^{60\times60}$: 可塑
    - $\mathbf W_{\mathrm{HR}\to\mathrm{HD}}\in\mathbb R^{60\times60}$: 可塑
    - $\mathbf W_{\mathrm{HD}\to\mathrm{HR}}\in\mathbb R^{60\times60}$: 固定 sparse selector
    - baseline 对 presynaptic drive 使用 `raw_sum`, 不乘 $1/N$

- external input
    - visual input 进入 HD proximal compartment
    - angular velocity input 进入 HR cells
    - 当前 wing 顺序为 $[\mathrm{LHR},\mathrm{RHR}]$: 

      $$
      \vec I_{\mathrm{vel}\to\mathrm{LHR}}=+k_vv\vec1,
      \qquad
      \vec I_{\mathrm{vel}\to\mathrm{RHR}}=-k_vv\vec1.
      $$

    - `visual.normalize_peak` 只定义 visual-current profile 的峰值, 不会归一化 $r$、$P$ 或 $E$

# Network dynamics

- HD cell 是 2-compartment neuron, 但当前实现包含三个独立电学状态: distal current、distal voltage 和 proximal voltage. 

  $$
  \tau_s\dot{\vec I}_{\mathrm{HD},d}
  =-\vec I_{\mathrm{HD},d}
  +\mathbf W_{\mathrm{HD}\to\mathrm{HD}}\vec r_{\mathrm{HD}}
  +\mathbf W_{\mathrm{HR}\to\mathrm{HD}}\vec r_{\mathrm{HR}}
  -b_{\mathrm{HD}}\vec1,
  $$

  $$
  \tau_{l,\mathrm{HD}}\dot{\vec V}_{\mathrm{HD},d}
  =-\vec V_{\mathrm{HD},d}+\vec I_{\mathrm{HD},d},
  $$

  $$
  C\dot{\vec V}_{\mathrm{HD},a}
  =-g_L\vec V_{\mathrm{HD},a}
  -g_D(\vec V_{\mathrm{HD},a}-\vec V_{\mathrm{HD},d})
  +\vec I_{\mathrm{vis}\to\mathrm{HD}}+\vec I_{\mathrm{exc}}.
  $$

    - TODO 中写作 $I_{\mathrm{HD}}$ 的第二个低通量实际是 $V_{\mathrm{HD},d}$；proximal visual current 不是独立积分状态. 
    - $V_{\mathrm{HD},d}\neq I_{\mathrm{HD},d}$. 
    - distal prediction 是关闭 proximal input 后的反事实稳态: 

      $$
      \vec V_{\mathrm{HD},ss}=p\vec V_{\mathrm{HD},d},
      \qquad
      p=\frac{g_D}{g_D+g_L}=\frac23.
      $$

    - $V_{\mathrm{HD},a}$ 只在初始化时放到一致稳态；训练和测试时都动态积分, 不会每步被设成 $V_{\mathrm{HD},ss}$. 

- HR cell 使用一个 HD-rate low-pass state, HR current 和 firing rate 随后代数计算: 

  $$
  \tau_{\mathrm{HD}\to\mathrm{HR}}
  \dot{\vec r}_{\mathrm{HD}\to\mathrm{HR}}^{LP}
  =-\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP}+\vec r_{\mathrm{HD}},
  $$

  $$
  \vec I_{\mathrm{HR}}
  =\mathbf W_{\mathrm{HD}\to\mathrm{HR}}
  \vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP}
  +\vec I_{\mathrm{vel}\to\mathrm{HR}}-b_{\mathrm{HR}}\vec1,
  \qquad
  \vec r_{\mathrm{HR}}=f(\vec I_{\mathrm{HR}}).
  $$

# Local learning rule

> Plastic synapse 的 postsynaptic neuron 必须是 HD cell；当前可塑 pathway 为 HD$\to$HD 和 HR$\to$HD. 

- postsynaptic prediction error

  $$
  \vec E_{\mathrm{HD}}
  =f(\vec V_{\mathrm{HD},a})-f(p\vec V_{\mathrm{HD},d}).
  $$

  两项都是实际 firing rate, `e_hd` 不除以 $f_{\max}$. 

- presynaptic PSP

  $$
  \begin{aligned}
  \tau_s\dot{\vec P}_{\mathrm{syn}} &=-\vec P_{\mathrm{syn}}+\vec r_{\mathrm{pre}},\\
  \tau_{l,\mathrm{HD}}\dot{\vec P}_{\mathrm{pre}} &=-\vec P_{\mathrm{pre}}+\vec P_{\mathrm{syn}}.
  \end{aligned}
  $$

  当前实现是两级滤波, 不是单一 $\tau_P$ 滤波. 

- plasticity induction 和 weight update

  $$
  \tau_\delta\dot{\boldsymbol\delta}_{\mathrm{pre}\to\mathrm{HD}}
  =-\boldsymbol\delta_{\mathrm{pre}\to\mathrm{HD}}
  +\vec E_{\mathrm{HD}}\vec P_{\mathrm{pre}}^\top,
  $$

  $$
  \dot{\mathbf W}_{\mathrm{pre}\to\mathrm{HD}}
  =\eta_{\mathrm{pre}\to\mathrm{HD}}
  \boldsymbol\delta_{\mathrm{pre}\to\mathrm{HD}}.
  $$

    - HD 和 HR pathway 使用同号规则；HR 项没有额外负号. 
    - baseline 没有 weight decay、clipping、symmetry/balance constraint, 也不清零 diagonal. 
    - 代码以秒计时；baseline 的 $\eta=50$ 只是把 release 的毫秒时间导数换算到秒, 没有 $f_{\max}^3$ scaling. 

# Decode head direction

- population vector average(PVA)

  $$
  z_{\mathrm{PVA}}
  =\sum_{i=1}^{N_{\mathrm{HD}}}
  r_{\mathrm{HD},i}e^{\mathrm i\theta_{\mathrm{HD},i}},
  \qquad
  \hat\theta_{\mathrm{HD}}=\arg z_{\mathrm{PVA}}.
  $$

    - PVA 使用 $r_{\mathrm{HD}}=f(V_{\mathrm{HD},a})$；不使用 distal prediction、HR rate、电流或 PCA state. 
    - 论文中的 $1/N_{\mathrm{HD}}$ 是正常数, 不改变相角, 因此代码省略它；这与 firing-rate normalization 无关. 
    - paired geometry 下, 直接对 60 个 cell 求和等价于先按 30 个方向求 pair sum: 

      $$
      A_k=\sum_{i:\theta_i=\phi_k}r_i,
      \qquad
      z_{\mathrm{PVA}}=\sum_k A_ke^{\mathrm i\phi_k}.
      $$

    - PVA vector strength

      $$
      \rho_{\mathrm{PVA}}
      =\frac{|\sum_i r_i e^{\mathrm i\theta_i}|}{\sum_i r_i}
      $$

      只用于判断解码可信度, 不反馈到动力学或学习. 
    - 均匀活动、对称双 bump 或总 firing rate 很小时, PVA 可能无定义. 
    - PVA 是 circular first moment；Clark overlap 是 template matching, 两者一般不逐点相等. 

# Toy model 设计

## Release code mapping

| `original/fly_rec.py` | 当前状态 | 含义 |
| --- | --- | --- |
| `Iden` | `i_hd_distal` | distal input current |
| `V` | `v_hd_distal` | distal voltage |
| `u` | `v_hd_proximal` | dynamic proximal voltage |
| `I_PSP`, `PSP` | `p_*_synaptic`, `p_*` | 两级 PSP |
| `x` | `r_hd_to_hr_lp` | HD$\to$HR low-pass rate |
| `w[:, :NHR]` | `w_hr_to_hd` | HR$\to$HD plastic block |
| `w[:, NHR:]` | `w_hd_to_hd` | HD$\to$HD plastic block |
| `Delta` | `delta_w_*` | low-pass plasticity induction |

原始权重文件缺失只影响载入论文已经训练好的矩阵；不影响复现状态方程、更新顺序和从随机权重重新训练. 

## One-timestep data stream

令 step 输入为 $S_n$, 输出为 $S_{n+1}$: 

1. 根据速度模型 (常速或者 OU process 生成的随机速度) 生成本步 $v_{n+1}$; 

   $$
   v_{n+1}=
   \begin{cases}
   v_{\mathrm{const}}, & \text{constant},\\[2mm]
   v_n+\dfrac{\Delta t}{\tau_v}(\mu_v-v_n)
   +\sigma_v\sqrt{\dfrac{2\Delta t}{\tau_v}}\,\xi_{n+1},
   \quad \xi_{n+1}\sim\mathcal N(0,1), & \text{OU}.
   \end{cases}
   $$

2. 更新 $\theta_{n+1}= (\theta_n+v_{n+1}\Delta t)\,\mathrm{mod}\,2\pi$, 

   $$
   \theta_{n+1}
   =\operatorname{wrap}(\theta_n+v_{n+1}\Delta t)
   =\big[(\theta_n+v_{n+1}\Delta t+\pi)\bmod 2\pi\big]-\pi.
   $$

3. 构造 visual/velocity input. 

   $$
   \Delta\theta_i^{n+1}
   =\operatorname{wrap}(\theta_{\mathrm{HD},i}-\theta_{n+1}),
   $$

   $$
   I_{\mathrm{HD},a,i}^{n+1}
   =
   \begin{cases}
   s_{\mathrm{prox}}
   \left[
   A_{\mathrm{vis}}
   e^{\kappa_{\mathrm{vis}}(\cos\Delta\theta_i^{n+1}-1)}
   -b_{\mathrm{vis}}+I_{\mathrm{exc}}
   \right], & \text{visual teacher on},\\[2mm]
   0, & \text{visual teacher off},
   \end{cases}
   $$

   $$
   \vec I_{\mathrm{vel}\to\mathrm{HR}}^{n+1}
   =
   \begin{bmatrix}
   +k_vv_{n+1}\vec 1_{N_{\mathrm{HR}}/2}\\
   -k_vv_{n+1}\vec 1_{N_{\mathrm{HR}}/2}
   \end{bmatrix}.
   $$

4. 用旧 $W^n$、$r_{\mathrm{HD}}^n$、$r_{\mathrm{HR}}^n$ 计算 HD distal drive. 

   $$
   \vec I_{\mathrm{HD}\leftarrow\mathrm{HD}}^{\,n}
   =\mathbf W_{\mathrm{HD}\to\mathrm{HD}}^{n}\vec r_{\mathrm{HD}}^{\,n},
   $$

   $$
   \vec I_{\mathrm{HD}\leftarrow\mathrm{LHR}}^{\,n}
   =\mathbf W_{\mathrm{LHR}\to\mathrm{HD}}^{n}\vec r_{\mathrm{LHR}}^{\,n},
   \qquad
   \vec I_{\mathrm{HD}\leftarrow\mathrm{RHR}}^{\,n}
   =\mathbf W_{\mathrm{RHR}\to\mathrm{HD}}^{n}\vec r_{\mathrm{RHR}}^{\,n},
   $$

   $$
   \vec D_{\mathrm{HD},d}^{\,n}
   =\vec I_{\mathrm{HD}\leftarrow\mathrm{HD}}^{\,n}
   +\vec I_{\mathrm{HD}\leftarrow\mathrm{LHR}}^{\,n}
   +\vec I_{\mathrm{HD}\leftarrow\mathrm{RHR}}^{\,n}
   -b_{\mathrm{HD}}\vec1.
   $$

5. 更新 $I_{\mathrm{HD},d}^{n+1}$, 再用新 $I_{\mathrm{HD},d}^{n+1}$ 更新 $V_{\mathrm{HD},d}^{n+1}$. 

   $$
   \vec I_{\mathrm{HD},d}^{\,n+1}
   =\vec I_{\mathrm{HD},d}^{\,n}
   +\frac{\Delta t}{\tau_s}
   \left(-\vec I_{\mathrm{HD},d}^{\,n}+\vec D_{\mathrm{HD},d}^{\,n}\right),
   $$

   $$
   \vec V_{\mathrm{HD},d}^{\,n+1}
   =\vec V_{\mathrm{HD},d}^{\,n}
   +\frac{\Delta t}{\tau_{l,\mathrm{HD}}}
   \left(-\vec V_{\mathrm{HD},d}^{\,n}
   +\vec I_{\mathrm{HD},d}^{\,n+1}\right).
   $$

6. PSP 第一级读取旧 presynaptic rates；第二级读取新 $P_{\mathrm{syn}}^{n+1}$, 得到 $P_{\mathrm{HD}}^{n+1}$ 和 $P_{\mathrm{HR}}^{n+1}$. 

   $$
   \vec P_{X,\mathrm{syn}}^{\,n+1}
   =\vec P_{X,\mathrm{syn}}^{\,n}
   +\frac{\Delta t}{\tau_s}
   \left(-\vec P_{X,\mathrm{syn}}^{\,n}+\vec r_X^{\,n}\right),
   \qquad X\in\{\mathrm{HD},\mathrm{HR}\},
   $$

   $$
   \vec P_X^{\,n+1}
   =\vec P_X^{\,n}
   +\frac{\Delta t}{\tau_{l,\mathrm{HD}}}
   \left(-\vec P_X^{\,n}+\vec P_{X,\mathrm{syn}}^{\,n+1}\right).
   $$

7. 用旧 $r_{\mathrm{HD}}^n$ 更新 $r_{\mathrm{HD}\to\mathrm{HR}}^{LP,n+1}$, 再得到 $I_{\mathrm{HR}}^{n+1}$ 和 $r_{\mathrm{HR}}^{n+1}$. 新 HR rate 不返回影响本步已经完成的 distal update. 

   $$
   \vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,n+1}
   =\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,n}
   +\frac{\Delta t}{\tau_{\mathrm{HD}\to\mathrm{HR}}}
   \left(-\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,n}
   +\vec r_{\mathrm{HD}}^{\,n}\right),
   $$

   $$
   \vec I_{\mathrm{HR}}^{\,n+1}
   =\mathbf W_{\mathrm{HD}\to\mathrm{HR}}
   \vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP,n+1}
   +\vec I_{\mathrm{vel}\to\mathrm{HR}}^{n+1}
   -b_{\mathrm{HR}}\vec1,
   \qquad
   \vec r_{\mathrm{HR}}^{\,n+1}=f(\vec I_{\mathrm{HR}}^{\,n+1}).
   $$

8. 用旧 $V_{\mathrm{HD},a}^{n}$、新 $V_{\mathrm{HD},d}^{n+1}$ 和本步 proximal input 更新 $V_{\mathrm{HD},a}^{n+1}$. 

   $$
   \vec V_{\mathrm{HD},a}^{\,n+1}
   =\vec V_{\mathrm{HD},a}^{\,n}
   +\frac{\Delta t}{C}
   \left[
   -g_L\vec V_{\mathrm{HD},a}^{\,n}
   -g_D\left(\vec V_{\mathrm{HD},a}^{\,n}
   -\vec V_{\mathrm{HD},d}^{\,n+1}\right)
   +\vec I_{\mathrm{HD},a}^{\,n+1}
   \right].
   $$

9. 计算 $r_{\mathrm{HD}}^{n+1}$、distal prediction 和 $E_{\mathrm{HD}}^{n+1}$. 

   $$
   \vec V_{\mathrm{HD},ss}^{\,n+1}
   =p_{d\to a}\vec V_{\mathrm{HD},d}^{\,n+1},
   \qquad
   \vec r_{\mathrm{HD}}^{\,n+1}
   =f(\vec V_{\mathrm{HD},a}^{\,n+1}),
   $$

   $$
   \vec r_{\mathrm{HD},d\text{-}\mathrm{pred}}^{\,n+1}
   =f(\vec V_{\mathrm{HD},ss}^{\,n+1}),
   \qquad
   \vec E_{\mathrm{HD}}^{\,n+1}
   =\vec r_{\mathrm{HD}}^{\,n+1}
   -\vec r_{\mathrm{HD},d\text{-}\mathrm{pred}}^{\,n+1}.
   $$

10. 训练时用新 error 和新 PSP 更新 $\delta^{n+1}$, 再从旧 $W^n$ 得到 $W^{n+1}$；测试时冻结 $\delta$ 和 $W$. 

    $$
    \boldsymbol\Pi_X^{\,n+1}
    =\vec E_{\mathrm{HD}}^{\,n+1}(\vec P_X^{\,n+1})^\top,
    \qquad
    X\in\{\mathrm{HD},\mathrm{HR}\},
    $$

    $$
    \boldsymbol\delta_X^{\,n+1}
    =\boldsymbol\delta_X^{\,n}
    +\frac{\Delta t}{\tau_\delta}
    \left(-\boldsymbol\delta_X^{\,n}+\boldsymbol\Pi_X^{\,n+1}\right),
    $$

    $$
    \widetilde{\mathbf W}_{X\to\mathrm{HD}}^{\,n+1}
    =\mathbf W_{X\to\mathrm{HD}}^{n}
    +\Delta t\,\eta_{X\to\mathrm{HD}}\boldsymbol\delta_X^{\,n+1},
    \qquad
    \mathbf W_{X\to\mathrm{HD}}^{n+1}
    =\mathcal C_X\!\left(
    \widetilde{\mathbf W}_{X\to\mathrm{HD}}^{\,n+1}
    \right),
    $$

    $$
    \text{testing:}\qquad
    \boldsymbol\delta_X^{\,n+1}=\boldsymbol\delta_X^{\,n},
    \qquad
    \mathbf W_{X\to\mathrm{HD}}^{n+1}
    =\mathbf W_{X\to\mathrm{HD}}^{n}.
    $$

11. $W^{n+1}$ 从下一个 timestep 才参与 distal drive. 

    $$
    \vec I_{\mathrm{HD},d}^{\,n+2}
    =\vec I_{\mathrm{HD},d}^{\,n+1}
    +\frac{\Delta t}{\tau_s}
    \left[
    -\vec I_{\mathrm{HD},d}^{\,n+1}
    +\mathbf W_{\mathrm{HD}\to\mathrm{HD}}^{n+1}\vec r_{\mathrm{HD}}^{\,n+1}
    +\mathbf W_{\mathrm{HR}\to\mathrm{HD}}^{n+1}\vec r_{\mathrm{HR}}^{\,n+1}
    -b_{\mathrm{HD}}\vec1
    \right].
    $$

因此当前实现是 **single-clock ordered Euler**: 每个全局 timestep 执行一次固定序列, 步内会使用刚更新的量. 它不是纯 Jacobi 同步更新, 也不是 event-driven asynchronous update. 权重从训练第一步起每步更新, 不等待电流或电压弛豫完成. 

## Slow-manifold state and PCA

- 冻结权重、关闭视觉、令速度为零时, 最小 Markov state 为

  $$
  x=[\vec r_{\mathrm{HD}\to\mathrm{HR}}^{LP},\vec r_{\mathrm{HR}},\vec I_{\mathrm{HD},d},\vec V_{\mathrm{HD},d},\vec V_{\mathrm{HD},a}],
  $$

  其维数为 $D=4N_{\mathrm{HD}}+N_{\mathrm{HR}}=300$. 

- 当前代码从真实 frozen discrete map $G_{\Delta t}$ 定义

  $$
  F_{\Delta t}(x)=\frac{G_{\Delta t}(x)-x}{\Delta t},
  \qquad
  q(x)=\frac12\|F_{\Delta t}(x)\|_2^2.
  $$

  $q$、flow 和 Jacobian 都在完整 300 维状态上计算；PVA 只提供 heading phase. 

- variance--rank diagnostic 对两套 observable 分别做 centered SVD: 
    - $R=[r_{\mathrm{HD}},r_{\mathrm{HR}}]$: PC1--3 图使用的 firing-rate state
    - $A\in\mathbb R^{M\times30}$: 进入 PVA complex sum 前的 paired-HD angular-rate statistic

  $$
  R_c=USV^\top,
  \qquad
  \gamma_k=\frac{s_k^2}{\sum_j s_j^2},
  \qquad
  C_K=\sum_{k=1}^{K}\gamma_k.
  $$

  `ramesan_pca_variance_rank.png` 显示单个与累计 explained variance, 并报告 80%、90%、95% rank 和 participation ratio. 若 $C_3\approx55\%$, PC1--3 只能作定性可视化；这说明三维线性投影不充分, 但不单独否定一维非线性闭环. 

## Numerical integration

- 论文和 release 使用 $\Delta t=0.5\,\mathrm{ms}$ forward Euler；论文没有报告 timestep convergence 或 stiff-solver comparison. 
- Eq. (4) 最快固定-drive 模态为 $C/(g_L+g_D)=1/3\,\mathrm{ms}$. 显式 Euler 要求

  $$
  \Delta t\frac{g_L+g_D}{C}<2.
  $$

  因此 $1\,\mathrm{ms}$ 不稳定；$0.5\,\mathrm{ms}$ 稳定但快模态会出现符号交替. 稳定不等于已经达到连续时间精度. 
- 选择 timestep 时, 应让 $0.5,0.25,0.125,0.0625\,\mathrm{ms}$ 使用同一初态和同一随机输入路径, 再比较最终权重结构、PVA、darkness PI 和 velocity gain. 选择满足科学误差阈值的最大 $\Delta t$. 
- `solve_ivp(BDF/Radau)` 适合做短时、确定性参考解, 不适合直接替换 $80{,}000\,\mathrm{s}$ 的随机在线训练循环. Diffrax 需要独立 JAX/SDE backend 和实测 benchmark. 
- release 对齐实验继续使用 `release_euler`；指数/ZOH 或其他 solver 应作为单独的 convergence 模式验证. 

### Block-multirate plasticity

- 当前完整学习状态不能写成导师草图中的
  $\tau_w\dot W=-W+\cdots$：论文/代码实际使用
  $\tau_\delta\dot\delta=-\delta+EP^\top$ 和 $\dot W=\eta\delta$，没有
  $-W$。联合 ODE 若加入该项会改变学习规则。
- `single_clock` 保留现有 ordered timestep。`block_multirate` 仍用相同
  $\Delta t$ 推进 neural、compartment 和 PSP 状态，但在 $K$ 个 microsteps
  内固定 $W$，保存每步 $E^n,P^n$，再在 block 边界累计可塑性。
- 令 $a=1-\Delta t/\tau_\delta$、$\Pi_j=E_jP_j^\top$，则一个含 $K$ 个
  samples 的 block 精确计算原 Euler 可塑性递推：

  $$
  \delta_K=a^K\delta_0+(1-a)\sum_{j=0}^{K-1}a^{K-1-j}\Pi_j,
  $$

  $$
  W_K=W_0+\Delta t\,\eta\left[
  \left(\sum_{m=1}^{K}a^m\right)\delta_0
  +\sum_{j=0}^{K-1}(1-a^{K-j})\Pi_j\right].
  $$

  两个加权 outer-product sums 由矩阵乘法完成。给定同一组 $E,P$ 时该结果
  与逐步 Euler 相同；近似只来自 block 内不把新 $W$ 反馈给神经动力学。
- 配置为 `simulation.training_integration_method: block_multirate` 与
  `plasticity_update_interval_duration: 0.01`。当前 baseline 显式保留
  `single_clock`；多速率方法通过 profile opt in。
- 50,000-step matched-stream 基准中，10 ms block 对 $N=60/120$ 分别为
  2.39x/2.73x；HD recurrent weight 的相对 Frobenius 差异约
  $3.8\times10^{-7}/1.1\times10^{-6}$。这些短时结果只验证实现，长期训练仍需
  multi-seed weight/learning-error/frozen-PI convergence。

## Training duration

- 当前学习规则没有 $-\lambda W$, 但这不保证逐点达到 $\dot W=0$. 
- 严格静止要求 $\delta=0$ 且 $E P^\top=0$；随机训练更合理的条件是窗口平均 $\langle\delta\rangle\approx0$. 
- Vafidis release 固定训练 $80{,}000\,\mathrm{s}$, 以低而稳定的 prediction error 和网络功能作为 convergence, 而不是要求每个突触严格停止. 
- weight norm 既不能证明收敛, 也不能单独证明未收敛. 至少同时检查: 
    - 论文式窗口 mean absolute error
    - 每突触绝对净漂移 $\|W(t)-W(t-T)\|_F/(T\sqrt{N_W})$
    - mean/RMS update 和 weight profile
    - 冻结权重后的 bump maintenance、darkness PI 和 velocity gain
- baseline 保持 `early_stopping: false`. 只有当绝对漂移、weight profile 和 held-out 功能指标都在多个窗口稳定时, 才使用研究性 early stopping. 

## Differences from the release file

- 当前工程用秒作为时间单位, 并使用可复现的分组件 RNG；release 文件使用毫秒且没有同样的 RNG stream. 
- 当前 HD $\to$ HR low-pass 使用标准 $\Delta t/\tau_s$ Euler；release 的 `x += (f-x)/tau_s` 字面漏写 `dt`, 当前实现遵循论文方程. 
- heterogeneous teacher、`presynaptic_population_mean`、clipping、symmetry 和 balance 都是显式可选实验；`vafidis_toy.yaml` baseline 使用 von Mises、`raw_sum` 和无结构约束. 

## 2026-08-16 diagnostic contract

- PI 的主误差以 darkness release 为零点，使用未包裹累计位移误差

  $$
  e_{PI}(t)=[\operatorname{unwrap}\hat\theta(t)-\hat\theta(0)]
  -[\operatorname{unwrap}\theta(t)-\theta(0)].
  $$

  因而 cue alignment offset 和多圈后的 $2\pi$ folding 不会混入 integrator
  error。wrapped error 仍保存，但只作圆周显示。

- constant velocity 同时报告 $e_{PI}(t)$、decoded velocity bias 和由 bias
  推算的长时漂移；OU ensemble 分开报告 mean drift、trial variance 与
  effective diffusion。短时 gain 接近 1 和长时相位误差很大可以同时成立。

- endpoint map 使用真实 autonomous release phase

  $$
  D(\phi)=\operatorname{wrap}(E_T(\phi)-\phi)
  $$

  的周期过零：$+\to-$ 为 stable，$-\to+$ 为 unstable。initial cue 只作为
  图的横坐标；cue transfer 的 orientation/coverage 单独作为有效性门槛。

- `numerical_convergence` 从高分辨率 cue settling 得到同一个 release
  Markov state，然后比较多个 $\Delta t$ 与 proximal method 的完整 ordered
  step。它不把 `exact_linear` 解释成全系统解析解，也不把 release 中漏乘
  `dt` 的 HD→HR 行为写回正式模型。

- `train_duration` 是 exposure/hard cap，与积分稳定性分开。snapshot
  selection 可显式选择未包裹 phase error 或 RMS velocity bias；在多 seed
  证据形成前不根据单次 norm 增长直接改学习率或 hard cap。

# Check

1. 短暂 visual pulse 后令 $v=0$, 检查 bump 是否在 darkness 中保持. 
2. 冻结权重并移除 visual teacher, 检查 $\hat\theta(t)\approx\theta(0)+\int_0^t v(t')\,\mathrm dt'$. 
3. 用不同常数 $v$ 拟合 $\dot{\hat\theta}\approx gv$, 理想结果为 $g\approx1$. 
4. 用 endpoint map 区分连续环与离散 basin. 
5. 用完整状态的 $q$、tangent/normal flow 和 Jacobian 检查 slow manifold；PCA 只负责可视化. 
