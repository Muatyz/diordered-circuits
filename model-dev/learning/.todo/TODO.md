1. 项目目标: 验证 Vafidis-style predictive local plasticity 是否能够学习 Head direction bump maintenance 和 (angular) velocity-driven path integration

2. 变量命名: 严格对应 notebook 中的物理量, 确保代码的可读性

3. data stream: 每个 timestep 需要按照固定顺序更新. 

    真实 $\theta, v$; visual/velocity input; HR dynamics; HD distal dynamics; proximal voltage; firing rate; learning error; PSP traces; weight update

4. 训练和测试分离: training phase 有 visual teacher 并 update weights; testing phase 冻结权重并且移除 visual teacher, 只保留短暂 cue 或直接完全 darkness

5. 数值积分方式: 先使用 Euler method 作为 baseline 实现. ODE 共用微分时间 dt, 不混用离散更新和连续时间公式

6. 诊断图与指标: 输出并保存 weight matrix, HD activity heatmap, decoded heading v.s. true heading, PI error, velocity gain curve, bump maintenance trace

7. 成功标准: 检查 $W_{HD\to HD}$ 是否形成 local symmetry, $W_{HR\to HD}$ 是否形成左右相反的不对称偏移结构, $\hat{\theta}$ 是否能在 darkness 下近似积分 $v(t)$

8. 该项目是验证 local learning rule 的, 因此严禁引入 back propagation, PyTorch autograd, global loss optimization, RNN trainer, supervised regression 等方法. 学习规则仅使用 Vafidis 论文中提到的局部变量方法. 


# 08.03 meeting notes

大多数都是针对代码库设计本身的, 如 stiffness of ODE 以及 solver, 从而确认计算的合理性

## 1

- [ ] Vafidis 的 2-compartment learning rule
  - [ ] 确认 $I_{HD}, I_{HD,d}$ 和 $W_{HD\to HD}$, $W_{HR\to HD}$ 的更新顺序, 是同步还是特定的异步
  - [ ] PVA 可视化算法的重新核定
    - [ ] 重新确认目前的 PVA 是针对什么 firing rate 完成的.目前的神经元因为 Vafidis' learning rule 被设计为 2-compartment, 因此动力学依赖严格来说是一个 2D 方程组 ($\dot{I}_{HD,d} = f_{1}(\cdot), \dot{I}_{HD} = f_{2}(\cdot)$), 在这种情况下如何定义 Ramesan 所使用的 $q = \frac{1}{2}||F(x)||^{2}$? 
    - [ ] 添加确认 PVA 的分解谱图示 (variance-rank 图), 确认一下取 $N = 3$ 的截断是否合理. 目前前三成分总和约为 55%

## 2

- [ ] 确认目前动力学方程的 stiffness, 确认目前的 Euler 法是否合理. 
  - 目前的 dt = [0.01, 0.001, 0.0005]s 的选择对于计算结果有影响吗? 是 dt 越小越好吗?
  - 导师推荐的一些数值积分方法, 是否有必要引入到目前的代码库中, 性能消耗和计算结果会更优越吗? 
    - [ ] Diffrax package;
    - [ ] `scipy.integrate.solve_ivp` 的 `method='BDF'` 或者 `method='Radau'`

- [ ] 确认训练终止的合理时间. 目前呈现的趋势是训练越长, weight norm 越大并且没有呈现出收敛趋势. 按照权重更新的动力学, 应该是会完全静止才对. 是否需要考虑对 weight norm 进行某种渐进行为分析? 
  - [x] 增加 frozen-weight snapshot PI development 诊断：按配置选择训练比例（默认每 1%），用对称 constant velocities 计算 time-averaged circular accumulated PI error。
  - [x] 80,000 s baseline 初测：最佳单点约 22,400 s；3--9 snapshot 平滑趋势最低点约 21,600--24,000 s；最终误差由 2.58° 恶化到 36.24°，同时 PVA strength 保持稳定，支持 overtraining 而非 bump 消失。
  - [x] 增加 Vafidis Eq. (19) absolute learning error 在线记录：默认 10 s forward window、每 1% 训练时间采样，保存 spikes/s population/per-neuron error，并输出到 `figures/diagnostics`。
  - [ ] 在下一次正式 80,000 s 训练中联合比较 absolute learning error、effective weight norm 与 frozen-weight PI error；旧 run 的稀疏 RMS history 无法精确重建 Eq. (19)。
  - [ ] 用多个 seed 与 fixed-seed OU ensemble 复核约 22--24 ks 的候选停止区间，再决定是否将其设为默认 hard cap / checkpoint patience 区间。

## 3

- [ ] 寻找更好的确认 attractor basin 的方法. 目前的 trajectories 追踪做得太糙了. 理论上来说 unstable 和 stable fixed point 应该是成对存在的

## 4

- [ ] 确认一下非线性函数具体参数对于 state space 的影响

- [ ] 重新确认速度场的计算方法是怎么完成的, 是不是自己描述的那样, 以及是否有更好的方法; 以及如何理解蓝线和橙色线的测定, 两者的差异是如何产生的


# 08.16

1. 目前的 PI error 仍然处于非常荒谬的状态, 无论是 constant velocity 还是 OU process, 得到的 error 都非常大. 这还是在 best weight 下诊断的结果. 这是否和 "HR->HD 的权重增长速度快于 HD->HD 的增长速度" 有关?

   - [x] PI 主指标统一为 cue-release-relative unwrapped accumulated error；constant/OU/snapshot 同时保存 wrapped companion、velocity bias、systematic drift 和 ensemble variance/diffusion。
   - [x] snapshot 诊断联合保存 HD→HD/HR→HD effective norm growth、HR/HD ratio，并在图中对齐 PI error；training history 图加入三条 realized pathway-current RMS。
   - [x] `vafidis_toy.yaml` 可直接运行目标诊断：训练不按行为阈值早停，结束后用多 heading / 多 velocity 的 RMS bias 离线排序 snapshot，并同时保存 final/best 权重。
   - [ ] norm ratio 本身不足以证明因果；需用多个 seed 联合比较 local learning error、pathway current 与 frozen PI，再决定是否修改学习率/归一化。

2. 目前的 trajectories 实验中, 一方面目前的实验非常失败, 另一方面目前的可视化非常臃肿, 而且也存在逻辑问题. 我认为应该按照这样的逻辑重新绘制: 1. 左边仍然沿用 trajectories 的绘制方式; 2. 中间绘制 initial cue angle 和 cue 充分弛豫后 angle 的对比图(理想情况下, 两者应当完全相同); 3. 右边绘制 endpoint map 和 initial cue angle 的对比图, 去除之前遗留的 cue on set/off 等复杂可视化逻辑. 然后在该基础上推算 stable/unstable fixed point. 需要注意推算的合理性, 因为理论上形成的慢环流形 stable 和 unstable 应当是成对出现的.

   - [x] canonical PVA 图改为 darkness trajectories / cue→release / cue→endpoint 三面板，停止生成旧的多-decoder cue onset/off 主图。
   - [x] fixed point 改为 actual release coordinate 上 endpoint displacement 的周期过零；处理 seam、cue-map orientation/coverage、unresolved gaps 和 alternation mismatch，不强制补齐成对结果。

3. 从数值计算的角度重新核验一遍目前代码的动力学实现是否存在问题, 特别是如果采取了 `exact_linear` 的计算方法, 是否仍然有必要继续沿用目前的 0.00025 和 50000 train duration. 之前虽然使用 1ms 计算并不严谨, 但是至少没有出现现在非常严重的 heading 无法 follow 的情况, 因此我高度怀疑目前的代码在数值计算上是存在问题的, 如果无法解决这个问题我无法安心实验并且相信其导出的结论

   - [x] 新增 common-runner `numerical_convergence`：从同一个高分辨率 cue-release state 比较 1/0.5/0.25/0.125 ms 与 high-resolution reference，保存 heading/rate/voltage/low-pass error。
   - [x] 明确 `exact_linear` 只精确推进 Eq. (4) proximal 子步；1 ms forward Euler 因 amplification=-2 被拒绝；release HD→HR 漏 `dt` 仅作为 parity metadata。
   - [x] baseline best-weight 2 s 初测表明 0.5/1 ms 不能由 `exact_linear` 自动视为收敛；0.25 ms exact 为边界结果，严格双阈值下 0.125 ms 才通过。详见 `reports/notes/2026-08-16_pi_trajectory_numerics_audit.md`。
   - [ ] 50,000 s 仍仅作 hard cap；用 matched-noise multi-seed training convergence 决定 timestep、checkpoint patience 和默认训练预算。


# 08.17

今天导师和我沟通了以下两个想法:

1. 区别于 Euler method,

  视觉输入相当于 $\frac{\mathrm{d}\vec{x}}{\mathrm{d}t} = f(\vec{r}, \theta)$ 的 sde, 神经活动相当于 $\tau\frac{\mathrm{d}\vec{r}}{\mathrm{d}t} = -\vec{r} + \cdots$, 权重更新相当于 $\tau_{w}\frac{\mathrm{d}W}{\mathrm{d}t} = -W + \cdots$. 这样就可以抽象出一个联合含时变量 $\vec{y} = [\vec{x}, \vec{r}, \vec{W}]$, 令其进行 ode 求解. 这样求解能够同时解决 Euler 不稳定性和求解速度慢的问题吗?

  - [x] 结论：不直接采用 full-state `solve_ivp(BDF/Radau)`。当前规则没有
    `$-W$`，且包含权重与 $\delta$ 后联合状态约 1.5 万维；最快 1/3 ms 模态、
    OU 随机输入和隐式 Jacobian 成本使其不适合作为长训练后端。
  - [x] 新增 opt-in `block_multirate`：神经状态保持原 `dt`，10 ms 内冻结慢
    权重，并用所有 microstep 的局部 `$EP^\top$` 样本在 block 边界代数累计
    原 `$\delta/W$` Euler 方程；`single_clock` 继续作为 release baseline。
  - [ ] 用完整训练预算、matched-noise 多 seed 比较两种方法的 learning error、
    weight profile/norm、best snapshot 与 frozen PI，再决定是否将多速率设为默认。


# 08.20 PI robustness follow-up

- [x] 训练中 checkpoint 与训练后 snapshot 统一使用 moving-cue 的 frozen
  heading/velocity grid，消除两套 “best” 语义不一致的问题。
- [x] 增加 worst-case velocity bias、zero drift、stall fraction，以及要求所有
  heading/正反方向通过的 depinning threshold；先筛 acceptance，再最小化 score，
  无通过项时明确记录 fallback。
- [x] 在不修改 Vafidis predictive local rule 的前提下，增加从宽速度到低速度的
  OU 标准差分阶段 schedule。
- [x] 增加可选 N=120 finite-size profile，仅按 release 规则缩放随机初始化，不施加
  symmetry/circulant constraint。
- [ ] 用 matched seed 比较原 N=60、PI-robust N=60 与 PI-robust N=120 的完整训练；
  重点比较 acceptance rate、worst-case bias、depinning speed、constant-velocity
  darkness error 和 OU ensemble drift/variance。
