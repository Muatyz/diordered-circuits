- [x] Figure 2 复现

    - [x] C: 经典 ring attractor 调谐曲线平移不变
    - [x] D: postsubicular 头朝向细胞的神经元调谐曲线具有异质性, 并不具有方向的平移不变性
    - 调谐曲线峰值对齐($\theta=0^{\circ}$)后, 小鼠 $k$ 在某角度 $\theta_{i}$ 会有 $N$ 个 $v_{j}(\theta_{i})$,
    - [x] E
        - [x] 浅色线: 单只小鼠 $k$ 对神经元求均值 $\begin{aligned} \langle v^{(k)}(\theta_{i})\rangle = \frac{1}{N}\sum_{j=1}^{N}v^{(k)}_{j}(\theta_{i}) \end{aligned}$
        - [x] 粗实线: 对 $M$ 只小鼠求均值 $\begin{aligned} \bar{v}(\theta) = \frac{1}{M}\sum_{k=1}^{M} \langle v^{(k)}(\theta)\rangle \end{aligned}$
    - [x] F
        - [x] 浅色线: 单只小鼠 $k$ 对神经元求标准差 $\begin{aligned}\sigma_{k}(\theta_{i}) = \sqrt{\frac{1}{N}\sum_{j=1}^{N}[v_{j}(\theta_{i})-\langle v_{k}(\theta_{i})\rangle]^{2}}\end{aligned}$
        - [x] 粗实线: 对 $M$ 只小鼠求标准差均值 $\begin{aligned}\bar{\sigma}(\theta) = \frac{1}{M}\sum_{k=1}^{M}\sigma_{k}(\theta) \end{aligned}$


- [ ] Figure 3 复现. $N=1533$ 个神经元
    - [ ] A: 具有不同偏好方向 $\theta$ 的神经元随机赋予编号 $i$, 按照最优化公式 $\begin{aligned}J_{ij}[\omega(t)] = \frac{1}{N}\int_{0}^{2\pi}\mathrm{d}\theta\int_{0}^{2\pi}\mathrm{d}\theta^{\prime}K_{i,\lambda}^{-1}(\theta,\theta^{\prime})\left[1 + \tau\omega(t)\frac{\partial}{\partial\theta}\right]x_{i}^{*}(\theta)\phi_{j}^{*}(\theta^{\prime})\end{aligned}$ 计算的 $J_{ij}$ 矩阵毫无规律
    - [ ] B: 仍是从 A 构造的 $J_{ij}$ 矩阵, 但是按照偏好方向顺序赋予 $i$, 观察到粗糙的 Mexican hat 对称阵, 以及连续吸引子结构
    - [ ] C: 求平均操作
        - [ ]  定义神经元有向间距 $k = (i-j)\,\mathrm{mod}\,N$ (周期性条件), 
        
            > 如 $k=0$ 为主对角线, $k=1$ 为主对角线下第一条线和右上角

        - [ ]  对距离 $k = i-j$ 上的 $N$ 个元素求平均值 $\begin{aligned}\bar{J}_{k} = \frac{1}{N}\sum_{i=0}^{N-1}J_{i,i-k}\end{aligned}$, 从而得到 $N$ 个平均值 $\{\bar{J}_{0}, \bar{J}_{1}, \cdots, \bar{J}_{N-1}\}$

        - [ ] 赋予 $\bar{J}_{ij} = \bar{J}_{(i-j)}$. 因为每一行都是上一行的向右平移, 因此 $\bar{J}_{ij}$ 是循环矩阵(circulant)

    - [ ] D: 在 C 的基础上加入噪声(并非随机 Gaussian 噪声)
        - [ ] 噪声矩阵 $R_{ij} = J_{ij} - \bar{J}_{ij}$
        - [ ] 对 $R_{ij}$ 进行随机重排, 得到 $R_{ij}^{\prime}$, 因此 $R_{ij}^{\prime}$ 的能量, 数值分布和 $R_{ij}$ 完全一致, 避免只使用 Gaussian 噪声的过分简化
        - [ ] 构造 $J_{ij}^{\prime} = \bar{J}_{ij} + R_{ij}^{\prime}$, 发现连续吸引子结构崩溃, 说明看似无规律的噪声实际上也是用于维护连续吸引子结构的必要成分

    - [ ] E: 计算目标流形 $\vec{x}^{*}$ 的三个主成分用于可视化. 
        - [ ] $\theta\in[0,2\pi]$ 对应颜色进行染色
        - [ ] $\times$: 初始条件, 在目标流形 $\vec{x}^{*}(\theta)$ 附近
        - [ ] $\cdot$: 经过 30 秒演化后, 从 $\times$ 演化 (由 A 或 B 中的 $J$ 支配, 这两者等价) 到目标流形 $\vec{x}^{*}(\theta)$ 上
    

    - [ ] F: 不同初始条件 ($\theta$) 演化中到目标流形的最近距离随时间 $t$ 变化. 均为收敛 ($<10^{-3}$)

    - [ ] G: 和 B 一样重新排好顺序的神经元 (COM order). 使用 $J_{ij}$ 支配的动力学演化到目标流形后的集群活动(放电率为指标的 bump)

    - [ ] H: 和 B 一样重新排好顺序的神经元 (COM order). 使用 $\bar{J}_{ij}$ 支配的动力学演化得到的 bump 活动(完美平移对称的曲线). 

    - [ ] I: 和 B 一样重新排好顺序的神经元 (COM order). 使用 $J_{ij}^{\prime}$ 支配的动力学演化得到的集群活动, 发现只收敛到了几个离散峰值的 bump 活动分布. 
        - [ ] 重排噪声 $R_{ij}\to R_{ij}^{\prime}$ 破坏了吸引子的连续性