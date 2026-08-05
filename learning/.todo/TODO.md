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

## 3

- [ ] 寻找更好的确认 attractor basin 的方法. 目前的 trajectories 追踪做得太糙了. 理论上来说 unstable 和 stable fixed point 应该是成对存在的

## 4

- [ ] 确认一下非线性函数具体参数对于 state space 的影响

- [ ] 重新确认速度场的计算方法是怎么完成的, 是不是自己描述的那样, 以及是否有更好的方法; 以及如何理解蓝线和橙色线的测定, 两者的差异是如何产生的