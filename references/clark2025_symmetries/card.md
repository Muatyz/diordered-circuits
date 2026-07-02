# Paper Card: clark2025_symmetries

## Bibliography

- Title: Symmetries and Continuous Attractors in Disordered Neural Circuits
- Authors: David G. Clark, L. F. Abbott, Haim Sompolinsky
- Year: 2025 (bioRxiv version posted 2025-10-26)
- DOI: https://doi.org/10.1101/2025.01.26.634933
- Local PDF: `paper.pdf`
- Search cache: `paper.txt`
- Status: reproduced / active foundation for new hypotheses

## One-sentence summary

实验 head-direction 调谐曲线虽高度异质，仍可构造出具有准连续吸引子动力学的 disordered RNN；其统计生成过程在大网络极限下恢复经典环吸引子的连续对称性、Mexican-hat 相互作用和谱简并。

## Why this paper matters

这是项目的 teacher model：定义目标流形、optimized recurrent weights、动力学验证、Gaussian generative process，以及后续 local learning rule 必须逼近的功能指标。

## Core claims

1. 数据导出的异质调谐流形可由最小范数 recurrent weights 稳定维持，并支持角速度积分。
2. 调谐曲线分布具有统计意义上的圆对称性，可由环上的 Gaussian process 与 normalized softplus 定量生成。
3. 看似无序的有限网络在 large-N 下等价于具有连续对称性的经典环吸引子；关键结构表现为 Fourier/spectral doublet degeneracy，而非逐元素循环对称。

## Mathematical objects

- `Phi = phi_star(theta)`: firing-rate manifold，shape `(N, n_theta)`
- `X = x_star(theta)`: input-current manifold，shape `(N, n_theta)`
- `J`: optimized recurrent matrix，shape `(N, N)`
- `Gamma_x(delta_theta)`: wrapped-Gaussian current covariance
- `C_phi(theta, theta')`: empirical rate correlation
- `m(theta, t)`: overlap order parameter
- `lambda`: ridge/minimum-norm regularization
- `c`: uniform inhibition stabilizing mean activity mode

## Implementation notes

- `softplus_inverse` 对近零 rate 极敏感；floor、平滑和归一化顺序必须记录。
- `lambda` 降低权重范数但增大 manifold flow residual；`c` 主要稳定 mean mode。
- 论文连续积分离散后，本项目统一使用 `Phi, X: (N, n_theta)` 和 `J: (N, N)`。
- 仅有低 flow residual 不足以证明吸引子稳定，必须检查扰动回归、切向漂移和 Jacobian。
- Fig. 3C/D 的 circulant 对照不是普通 Gaussian 噪声：应重排原 residual 以保持其幅度分布。

## Current project conclusion

Figure 2–6 已形成数据、图和诊断闭环；下一阶段应将 `J_opt` 视为 teacher/baseline，评价 local rule 是否恢复 manifold、normal stability、slow tangential drift 与谱结构，而不是只比较权重逐元素误差。
