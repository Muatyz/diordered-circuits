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


- [x] Figure 3 复现. $N=1533$ 个神经元
    - [x] A: 具有不同偏好方向 $\theta$ 的神经元随机赋予编号 $i$, 按照最优化公式 $\begin{aligned}J_{ij}[\omega(t)] = \frac{1}{N}\int_{0}^{2\pi}\mathrm{d}\theta\int_{0}^{2\pi}\mathrm{d}\theta^{\prime}K_{i,\lambda}^{-1}(\theta,\theta^{\prime})\left[1 + \tau\omega(t)\frac{\partial}{\partial\theta}\right]x_{i}^{*}(\theta)\phi_{j}^{*}(\theta^{\prime})\end{aligned}$ 计算的 $J_{ij}$ 矩阵毫无规律
    - [x] B: 仍是从 A 构造的 $J_{ij}$ 矩阵, 但是按照偏好方向顺序赋予 $i$, 观察到粗糙的 Mexican hat 对称阵, 以及连续吸引子结构
    - [x] C: 求平均操作
        - [ ]  定义神经元有向间距 $k = (i-j)\,\mathrm{mod}\,N$ (周期性条件), 
        
            > 如 $k=0$ 为主对角线, $k=1$ 为主对角线下第一条线和右上角

        - [ ]  对距离 $k = i-j$ 上的 $N$ 个元素求平均值 $\begin{aligned}\bar{J}_{k} = \frac{1}{N}\sum_{i=0}^{N-1}J_{i,i-k}\end{aligned}$, 从而得到 $N$ 个平均值 $\{\bar{J}_{0}, \bar{J}_{1}, \cdots, \bar{J}_{N-1}\}$

        - [ ] 赋予 $\bar{J}_{ij} = \bar{J}_{(i-j)}$. 因为每一行都是上一行的向右平移, 因此 $\bar{J}_{ij}$ 是循环矩阵(circulant)

    - [x] D: 在 C 的基础上加入噪声(并非随机 Gaussian 噪声)
        - [ ] 噪声矩阵 $R_{ij} = J_{ij} - \bar{J}_{ij}$
        - [ ] 对 $R_{ij}$ 进行随机重排, 得到 $R_{ij}^{\prime}$, 因此 $R_{ij}^{\prime}$ 的能量, 数值分布和 $R_{ij}$ 完全一致, 避免只使用 Gaussian 噪声的过分简化
        - [ ] 构造 $J_{ij}^{\prime} = \bar{J}_{ij} + R_{ij}^{\prime}$, 发现连续吸引子结构崩溃, 说明看似无规律的噪声实际上也是用于维护连续吸引子结构的必要成分

    - [x] E: 计算目标流形 $\vec{x}^{*}$ 的三个主成分用于可视化. 
        - [ ] $\theta\in[0,2\pi]$ 对应颜色进行染色
        - [ ] $\times$: 初始条件, 在目标流形 $\vec{x}^{*}(\theta)$ 附近
        - [ ] $\cdot$: 经过 30 秒演化后, 从 $\times$ 演化 (由 A 或 B 中的 $J$ 支配, 这两者等价) 到目标流形 $\vec{x}^{*}(\theta)$ 上
    

    - [x] F: 不同初始条件 ($\theta$) 演化中到目标流形的最近距离随时间 $t$ 变化. 均为收敛 ($<10^{-3}$)

    - [x] G: 和 B 一样重新排好顺序的神经元 (COM order). 使用 $J_{ij}$ 支配的动力学演化到目标流形后的集群活动(放电率为指标的 bump), 不同颜色代表不同的 COM ($\theta\in [0,2\pi]$). 

    - [x] H: 和 B 一样重新排好顺序的神经元 (COM order). 使用 $\bar{J}_{ij}$ 支配的动力学演化得到的 bump 活动(完美平移对称的曲线). 不同颜色代表不同的 COM ($\theta\in [0,2\pi]$). 

    - [x] I: 和 B 一样重新排好顺序的神经元 (COM order). 使用 $J_{ij}^{\prime}$ 支配的动力学演化得到的集群活动, 发现只收敛到了几个离散 COM 位置的 bump 活动分布. 不同颜色代表不同的 COM ($\theta\in [0,2\pi]$). 
        - [x] 重排噪声 $R_{ij}\to R_{ij}^{\prime}$ 破坏了吸引子的连续性


    - [x] J: 设均匀角速度 $\omega = 0.8\mathrm{rad/s}$ 下的重叠序参量(overlap order parameter) $\begin{aligned}m(\theta, t) = \frac{1}{N}\sum_{i=1}^{N}\phi_{i}^{*}(\theta)\phi_{i}(t)\end{aligned}$
        - 表征神经元放电率 $\phi_{i}(t)$ 与目标流形 $\phi_{i}^{*}(\theta)$ 的对齐程度
        - 网络活动处于流形坐标 $\varphi$ 时($\phi_{i}(t) = \phi_{i}^{*}(\varphi)$), 则 $m(\theta, t) = C^{\phi}(\theta, \varphi)$, 即关联函数
    - [x] K: 设真实小鼠角速度 $\omega(t)$, 计算模型预测的 $m(\theta, t)$
        - 模型对更复杂的含时信号 $\omega(t)$ 仍然具有积分追踪能力
    - [x] L: 根据真实记录的小鼠神经活动计算实验值 $m(\theta, t)$, time bin 为 100 ms

- [x] 图 4

    - [x] A: 不同小鼠(纵轴) 采样神经元的 COM 位置/偏好方向 $\{\theta\}$, 是真实实验数据
    - [x] B: 不同小鼠(纵轴) 采样神经元的 COM 位置/偏好方向 $\{\theta\}$, 根据设定算法生成的虚假数据, 偏好方向 $\theta$ 在区间 $[0,2\pi]$ 均匀随机分布, 并且各小鼠神经元数量和真实数据完全一致
    - [x] C: 随着记录的真实神经元数据子集 $N_{\mathrm{sub}}$ 增加 ($5\to 80$), 根据真实的小鼠实验数据计算的相关函数 $C^{\phi}(\theta,\theta^{\prime})$ 越来越趋近于循环矩阵 [$C^{\phi}(\theta,\theta^{\prime})\approx C^{\phi}(\theta-\theta^{\prime})$]

- [x] 图 5 (重点)

    - [x] A: $\begin{aligned}\frac{1}{\sigma}\end{aligned}$ 控制 $\Gamma^{x}(\Delta\theta)$ 在角空间的宽度
    - [x] B: $\sigma$ 越高, 采样得到的 $x(\theta)$ 振荡越剧烈
    - [x] C: $\beta$ 和 $b$ 对激活函数的影响. 生成的输入电流具有 Gaussian 分布
        - $\beta$: 越大, 激活函数越陡峭
        - $b$: 越大, 激活函数越往右平移. 控制放电率的稀疏性
    - [x] D & E: 如何根据记录的实验数据反推生成过程的参数 $(\sigma, \beta, b)$
        1. 根据实验数据计算 $\hat{\Gamma}_{n}^{\mathrm{data}}$, 当前复现取前 30 个非零 Fourier 系数
        2. 参数空间 $(\sigma, \beta, b)$ 控制生成过程得到的 $\hat{\Gamma}_{n}^{\mathrm{gen}}(\sigma, \beta, b)$
        3. 定义误差函数 $\begin{aligned}E(\sigma,\beta,b) = \sum_{n=1}^{30}\left[\hat{\Gamma}_{n}^{\mathrm{data}} - \hat{\Gamma}_{n}^{\mathrm{gen}}(\sigma, \beta, b)\right]^2\end{aligned}$
        4. D: 令 $\sigma$ 变化, 观察 $\underset{b,\beta}{\min} E(\sigma, \beta, b)$ 的变化. 发现 $\sigma\approx 1.42$ 存在全局最小值
        5. E: 对于固定 $\sigma_{0}$, 在 $(\beta, b)$ 空间中的 $E(\sigma_{0}, \beta, b)$ 热图. 颜色最深处(标记白点)为最小值 $\underset{\beta, b}{\min} E(\sigma_{0}, \beta, b)$
        - 当前 1533 个 QC 后神经元的复现结果: $\sigma_c=1.40$, $\beta_c\approx2.61$, $b_c\approx2.08$；原文报告值 $(1.42,2.76,1.73)$ 仍处于低误差谷内
    - [x] F: 比较根据 $\underset{\sigma, \beta, b}{\min} E(\sigma, \beta, b)$ 确定的最优参数 $(\sigma_{c}, \beta_{c}, b_{c})$ 生成的 $\Gamma^{\mathrm{gen}}(\Delta\theta)$ 和实验数据中的 $\Gamma^{\mathrm{data}}(\Delta\theta)$, 完全一致
        - 使用 100,000 条生成曲线估计 $\Gamma^{\mathrm{gen}}$, 与 1533 个实验神经元所得曲线的 RMSE 为 0.0121
        - 已提供完整批量生成与流式 batch 接口, 可保存包含 $\theta$, $x^{*}(\theta)$ 和 $\phi^{*}(\theta)$ 的假设性数据集供 `/learning` 使用
    - [x] G: 即使 $(\sigma_{c},\beta_{c},b_{c})$ 不变, 生成的微观层面的 $x^{*}(\theta)$ 仍然是各不相同的, 和真实的实验数据几乎无法分辨
        - 原图纵轴实际展示 normalized softplus 后的 firing rate $\phi^{*}(\theta)$；复现数据同时保存对应的 latent current $x^{*}(\theta)$
        - 使用相同最优参数独立生成 16 条曲线，得到单峰、多峰、不同峰宽及跨 $0/2\pi$ 边界等微观异质性

- [ ] 图 6 (诊断手法)
    生成过程的采样数 $N_{\mathrm{samples}} = 10^{5}$, 实验数据采样数 $N_{\mathrm{data}} = 1533$

    - [x] A: 生成过程预测的平均调谐曲线, 和单小鼠 $\begin{aligned}\langle v^{(k)}(\theta_{i})\rangle = \frac{1}{N}\sum_{j=1}^{N}v_{j}^{(k)}(\theta_{i})\end{aligned}$ 以及对 $M$ 只小鼠平均 $\begin{aligned}\bar{v}(\theta) = \frac{1}{M}\sum_{k=1}^{M}\langle v^{(k)}(\theta)\rangle\end{aligned}$ 进行比较
    - [x] B: 生成过程预测的对平均调谐曲线的标准差, 和单小鼠对调谐曲线标准差 $\begin{aligned}\sigma_{k}(\theta_{i}) = \sqrt{\frac{1}{N}\sum_{j=1}^{N}\big[v_{j}(\theta_{i}) - \langle v^{(k)}(\theta_{i})\rangle\big]^{2}}\end{aligned}$ 以及对 $M$ 只小鼠求标准差 $\begin{aligned}\bar{\sigma}(\theta) = \frac{1}{M}\sum_{k=1}^{M}\sigma_{k}(\theta)\end{aligned}$ 进行比较
    - [x] C: 设定峰检测阈值 $z_{\mathrm{thresh}} = [1.0, 0.5, 0.0]$, 用于统计调谐曲线 $v_{i}^{*}(\theta)$ 在 $[0,2\pi]$ 上有多少个峰
        - $z$-score 的计算. 调谐曲线为 $v_{i}(\theta)$
            1. 计算平均放电率 $\begin{aligned}\langle v_{i}\rangle = \frac{1}{2\pi}\int_{0}^{2\pi}v_{i}(\theta)\mathrm{d}\theta\end{aligned}$
            2. 计算放电率标准差 $\begin{aligned}\sigma_{i} = \sqrt{\frac{1}{2\pi}\int_{0}^{2\pi}\big[v_{i}(\theta)-\langle v_{i}\rangle\big]^{2}\mathrm{d}\theta}\end{aligned}$
            3. 寻找 $v_{i}(\theta)$ 的局部峰值 $h_{i,l}$ ($l=1,2,\cdots,N_{\mathrm{peak}}$)
            4. 计算各峰 $z$-score $\begin{aligned}z_{i,l} = \frac{h_{i,l} - \langle v_{i}\rangle}{\sigma_{i}}\end{aligned}$
        - 当 $z_{i,l} > z_{\mathrm{thresh}}$, 该峰计入统计
        - 灰色: 真实实验. 峰数量以及对应的神经元比例
        - 红色: 生成过程, 峰数量以及对应的神经元比例
    - [x] D: 调谐曲线 $v_{i}^{*}(\theta)$ 峰值的密度分布. 
        - 对于神经元 $i$, 记录到的峰高度为 $h_{i,1}>h_{i,2}>\cdots>h_{i,N_{\mathrm{peak}}}$, 集合为 $\{h_{i,l}\}$
        - 统计所有神经元的第一峰 $\{h_{1,1}, h_{2,1},\cdots,h_{N,1}\}$, 其能形成 peak 1 的频率分布(密度)
        - 对于 peak 2 和 peak 3 同理, 分别是 $\{h_{1,2}, h_{2,2},\cdots, h_{N,2}\}$ 和 $\{h_{1,3}, h_{2,3},\cdots, h_{N,3}\}$ 的密度分布
        - 因此不同神经元的 peak 1/2/3 可以存在简并, 如 $h_{i,1} = h_{j, 1}$, 使 peak 1 在这个 peak height 具有更高的频次
        - 原本应为 Gaussian 分布, 由于非线性激活和归一化使得分布呈现长尾
    - [x] E: 调谐曲线相对于 circular center of mass 的镜像对称性
        - 对小鼠 $k$ 的神经元 $i$, 其平滑并归一化后的调谐曲线记为 $v_{i}^{(k)}(\theta)$. 原文并不是围绕 peak 1 做镜像, 而是围绕该调谐曲线的 circular center of mass 做镜像
        - 定义 circular center of mass 角度 $\theta_{\mathrm{COM},i}^{(k)}$ 为

            $$
            \begin{align*}
            \theta_{\mathrm{COM},i}^{(k)} = \arg\left[\sum_{j=1}^{N_{\theta}} v_{i}^{(k)}(\theta_{j})e^{\mathrm{i}\theta_{j}}\right]
            \end{align*}
            $$

            其中 $N_{\theta}=100$ 是实验数据处理中使用的角度 bin 数. 若用连续变量表示, 则对应 $\theta_{\mathrm{COM},i}^{(k)} = \arg\left[\int_{0}^{2\pi}v_{i}^{(k)}(\theta)e^{\mathrm{i}\theta}\mathrm{d}\theta\right]$
        - 围绕 $\theta_{\mathrm{COM},i}^{(k)}$ 的环上反射定义为

            $$
            \begin{align*}
            \widetilde{v}_{i}^{(k)}(\theta) = v_{i}^{(k)}\left(2\theta_{\mathrm{COM},i}^{(k)}-\theta\right), \quad \theta \in [0,2\pi) \ \mathrm{mod}\ 2\pi
            \end{align*}
            $$

            如果先将每条曲线平移到 $\theta_{\mathrm{COM},i}^{(k)}=0$, 上式才等价于 $\widetilde{v}_{i}^{(k)}(\theta)=v_{i}^{(k)}(-\theta)$

        - 定义 $v_{i}^{(k)}(\theta)$ 和 $\widetilde{v}_{i}^{(k)}(\theta)$ 的翻转(flip)对称性为 Pearson 相关系数 
    
            $$
            \begin{align*}
            \rho_{i}^{(k),\mathrm{flip}} = \frac{\begin{aligned}
               \sum_{j=1}^{N_{\theta}}(v_{i}^{(k)}(\theta_{j}) - \langle v_{i}^{(k)}\rangle)(\widetilde{v}_{i}^{(k)}(\theta_{j}) - \langle \widetilde{v}_{i}^{(k)}\rangle)
            \end{aligned}}{\begin{aligned}
                \sqrt{\sum_{j=1}^{N_{\theta}}(v_{i}^{(k)}(\theta_{j}) - \langle v_{i}^{(k)}\rangle)^{2}}\sqrt{\sum_{j=1}^{N_{\theta}}(\widetilde{v}_{i}^{(k)}(\theta_{j}) - \langle \widetilde{v}_{i}^{(k)}\rangle)^{2}}
            \end{aligned}} = \frac{\mathrm{Cov}\left[v_{i}^{(k)}(\theta), \widetilde{v}_{i}^{(k)}(\theta)\right]}{\sqrt{\mathrm{Var}\left[v_{i}^{(k)}(\theta)\right] \mathrm{Var}\left[\widetilde{v}_{i}^{(k)}(\theta)\right]}}
            \end{align*}
            $$

        - 对所有小鼠 $k=\{1,2,\cdots,M\}$ 的所有神经元 $i=\{1,2,\cdots, N_{k}\}$ 都计算 $\rho_{i}^{(k),\mathrm{flip}}$ 并对其进行直方图统计(总共 $\begin{aligned}N_{\mathrm{data}} = \sum_{k=1}^{M}N_{k} = 1533\end{aligned}$ 个神经元), 得到其分布 $P(\rho^{\mathrm{flip}})$. 对生成过程采样得到的调谐曲线重复相同操作, 得到红线
        - $\rho_{i}^{(k),\mathrm{flip}}\approx 1$ 表示该神经元的调谐曲线围绕自身 circular center of mass 近似镜像对称; $\rho_{i}^{(k),\mathrm{flip}}$ 越小, 表明调谐曲线越偏离这种反射对称性
        - 这里检验的不是每条单神经元曲线是否严格等于某个确定的对称均值曲线, 而是实验数据中单细胞调谐曲线的非对称性分布能否被同一个统计生成过程复现. 原文的结果是: 生成过程不仅复现 $P(\rho^{\mathrm{flip}})$ 的主体, 也较好复现了强非对称调谐曲线对应的低 $\rho^{\mathrm{flip}}$ 拖尾
        - **结论: 单个神经元的调谐曲线可以明显偏离围绕 circular center of mass 的镜像对称性; 这种偏离不需要额外引入特定的不对称生物学机制, 而可以由具有 circular symmetry 的统计生成过程在单样本层面的涨落和非线性变换共同产生**
    - [x] F: 空间编码的信息熵容量
        - 小鼠朝向各角度 $\theta$ 的先验概率 $\begin{aligned}p(\theta) = \frac{1}{2\pi}\end{aligned}$
        - 小鼠 $k$ 的神经元 $i$ 的平均放电率 $\begin{aligned}\langle v_{i}^{(k)}\rangle = \frac{1}{2\pi}\int_{0}^{2\pi}v_{i}^{(k)}(\theta)\mathrm{d}\theta\end{aligned}$
        - 小鼠 $k$ 的神经元 $i$ 的调谐曲线 $v_{i}^{(k)}(\theta)$ 的互信息 $\begin{aligned}I_{\mathrm{HD},i}^{(k)} = \int_{0}^{2\pi}p(\theta)\frac{v_{i}^{(k)}(\theta)}{\langle v_{i}^{(k)}\rangle}\log_{2}\left(\frac{v_{i}^{(k)}(\theta)}{\langle v_{i}^{(k)}\rangle}\right)\mathrm{d}\theta\end{aligned}$
        
            > 神经元编码信息的能力取决于 $v(\theta)$ 相比于 $\langle v\rangle$ 的波动程度. 峰值越高, 每个脉冲携带的信息量越大
        - 对所有小鼠 $k$ 的所有神经元 $i$ 计算 $I_{\mathrm{HD},i}^{(k)}$ 并对其进行直方图统计, 得到分布 $P(I_{\mathrm{HD}})$, 生成过程的处理类似从而得到红线
        - **结论: Gaussian 生成过程和非线性激活函数的组合能够预测实验数据的信息熵分布**

Advice from ChatGPT: 

我的判断：**可以视为 Clark Fig. 3A–F 的“第一版/功能性复现成功”，但还不能说是“严格定量复现完成”。**

你现在已经完成了 Fig. 3A–F 的核心闭环：从 DANDI000939 提取 tuning curves，构造 target manifold，计算 optimized weight matrix，并且从微扰初态回到目标流形附近。原文 Fig. 3A–F 的关键不是单纯画出矩阵，而是证明 data-derived optimized weights 虽然看似 disorder，却能产生 quasi-continuous-attractor dynamics；原文标准是微扰初态在约 1 s 内快速回到流形附近，并在 30 s 内保持在流形附近。 你现在的 Fig. F 已经表现出大多数轨迹从约 (10^{-1}) 收敛到 (10^{-2}) 量级，这说明“优化权重确实产生了吸引到目标流形的法向稳定性”。

但它还不是严格复现，原因有三点。第一，你图里标注的是 (\lambda=10^{-4})，而原文用于 data-derived network dynamics 的参数是 (\lambda=10^{-6}) 和 (c=1)。 更强 regularization 很可能会牺牲 manifold flow residual，导致最终距离 floor 比原文高。第二，你的 Fig. F 里还有个别轨迹在 5–15 s 出现明显离开流形的膨胀，然后才缓慢回落，这说明 quasi-continuous attractor 的 timescale separation 还没有原文那么干净。第三，你现在展示的是 A–F，但原文 Fig. 3 后半部分还用 G–I 验证 optimized weights、circulant weights、circulant-plus-shuffled-residual weights 的动力学差异，并用 J–L 的 overlap order parameter 展示速度积分和真实数据读出。

所以最准确的表述是：

“我已经复现了 Fig. 3A–F 的主要构造流程和定性动力学现象：optimized (J) 在 random order 下无明显结构，在 COM order 下显露近似 Mexican-hat band；微扰初态能够被吸引回 target-current manifold 附近，距离下降到 (10^{-2}) 量级。但参数和收敛精度尚未完全达到原文设置，因此属于功能性复现，而非完全定量复现。”

接下来我建议：**不要完整卡在 overlap order parameter 全部复现上，也不要完全跳过它。先做一个最小版 overlap readout，然后立刻进入 generative process。**

原因是，overlap order parameter

[
m(\theta,t)=\frac{1}{N}\sum_i \phi_i^\star(\theta)\phi_i(t)
]

不是普通附图，而是把高维 RNN state 映射回低维环坐标的关键 readout。原文明确用它来表征网络活动和目标流形的对齐关系；如果网络状态位于 (\psi) 处的 target manifold 上，那么 (m(\theta,t)=C^\phi(\theta,\psi))，表现为以 (\psi) 为中心的 bump。 这对你之后做 local learning 特别重要，因为你不能只看 (||x(t)-x^\star(\theta)||)，还要知道网络是否真的保持了一个可解码的环坐标、是否发生了 tangential drift、是否 collapse 到离散点。

但是没有必要现在完整复现 Fig. 3J–L。你现在只需要补一个最小 diagnostic：

[
\hat{\theta}(t)=\arg\max_\theta m(\theta,t),
]

然后画两张图即可：一张是无速度输入、微扰初态后的 (m(\theta,t)) heatmap，看 bump 是否稳定停在原始角度附近；另一张是不同初态的 (\hat{\theta}(t))，看 30 s 内是否只发生缓慢 drift，而不是突然跳到几个固定 attractor。做到这一步，就足够把当前复现封版为 “Fig. 3A–F + minimal overlap validation”。

然后应当优先转向 **generative process**。这是你后续课题最需要的部分。原文自己也说，实验数据只有 (N_{\text{data}}=1533)，单个 mouse 更只有 (O(10^2)) neurons，因此需要 generative process 来采样 synthetic tuning curves，进而构造更大规模网络和做 large-(N) 分析。 这个 generative process 的核心就是从 circularly symmetric Gaussian process 采样 (x^\star(\theta))，再经过 normalized softplus 得到 (\phi^\star(\theta))。 对你现在的问题——“是否存在 local & unsupervised learning rule 逼近 optimized (J)”——synthetic data 比完整复现 Fig. 3J–L 更关键，因为它能让你系统扫描 (N,\sigma,\beta,b,\lambda,c)，并且构造可重复、可控、可扩展的训练样本。

我建议你的下一步路线按这个顺序走：

第一步，把当前 Fig. 3A–F 结果固定成一个 reproducible checkpoint。记录清楚 (N=1533)、bin 数、smoothing 方法、activation 参数、(\lambda)、(c)、dt、初态扰动强度、distance 定义、随机种子。尤其要明确写出“当前结果使用 (\lambda=10^{-4})，不是原文 (\lambda=10^{-6})”。

第二步，补最小 overlap order parameter。只做 (m(\theta,t))、(\arg\max_\theta m)、bump width、decoded phase drift。不要现在投入太多时间做真实 mouse trajectory 的 Fig. 3K/L。

第三步，进入 generative process。先用 Fourier 方法生成 circular GP，而不是直接大矩阵 Cholesky；然后实现 normalized softplus；最后用 synthetic (\phi^\star(\theta)) 重复 optimized (J) 与微扰收敛测试。先从 (N=512,1024,2048) 做，不要一上来追大规模。

第四步，再回头补 Fig. 3G–I：optimized (J)、diagonal-averaged circulant (J)、circulant plus shuffled residuals 三者的最终 activity pattern 对比。这个验证很重要，因为它能证明“disorder 不是普通噪声，而是 structured disorder”。
